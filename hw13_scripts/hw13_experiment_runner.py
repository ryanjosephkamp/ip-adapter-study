"""
HW13 Experiment Runner — Pipeline-specific generation and inference functions.
Each function wraps a specific pipeline with standardized parameter handling,
seed control, timing, and CSV logging.

Covers both vision (IP-Adapter) and audio (ASR, TTS) experiments.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - exercised only in missing-dependency environments
    torch = None

from hw13_data_utils import (
    append_csv_row,
    compute_wer_cer,
    get_audio_stats,
    now_iso,
    save_audio,
    save_image,
    timer,
)


def _require_torch():
    """Return torch if available, otherwise raise a clear dependency error."""
    if torch is None:
        raise ImportError("torch is required for HW13 experiment execution.")
    return torch


def _resolve_image_name(default_name, seed, metadata):
    """Build a deterministic image filename, allowing explicit notebook/orchestrator control."""
    filename_stem = metadata.get("filename_stem")
    if filename_stem:
        return f"{filename_stem}_seed{seed}.png"
    return default_name


def make_generator(seed, device="cuda"):
    """Create a seeded torch.Generator for reproducibility."""
    torch_module = _require_torch()
    resolved_device = device
    if device == "cuda" and not torch_module.cuda.is_available():
        resolved_device = "cpu"
    return torch_module.Generator(device=resolved_device).manual_seed(seed)


def run_ga18b(
    pipeline,
    prompt,
    reference_image,
    ip_adapter_scale,
    seed,
    num_inference_steps,
    save_dir,
    csv_path=None,
    sub_experiment="",
    extra_meta=None,
):
    """Run a single GA18B-style image variation with uniform IP-Adapter scale."""
    metadata = dict(extra_meta or {})
    save_dir = Path(save_dir)
    csv_path = Path(csv_path) if csv_path else None

    pipeline.set_ip_adapter_scale(ip_adapter_scale)
    generator = make_generator(seed)

    with timer(f"GA18B scale={ip_adapter_scale} seed={seed}") as timing:
        result = pipeline(
            prompt=prompt,
            ip_adapter_image=reference_image,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]

    default_name = f"ga18b_{sub_experiment}_scale_{ip_adapter_scale:.2f}_seed{seed}.png"
    image_name = _resolve_image_name(default_name, seed, metadata)
    image_path = save_image(result, save_dir / image_name, label=image_name)

    timestamp = now_iso()
    if csv_path is not None:
        row = [
            metadata.get("experiment", "exp2"),
            sub_experiment,
            "GA18B",
            seed,
            prompt,
            metadata.get("reference_image", "items_variation"),
            ip_adapter_scale,
            "",
            num_inference_steps,
            str(image_path.name),
            timing["elapsed_sec"],
            timestamp,
        ]
        append_csv_row(csv_path, row)

    return {
        "image": result,
        "image_path": image_path,
        "elapsed_sec": timing["elapsed_sec"],
        "timestamp": timestamp,
    }


def run_ga18c(
    pipeline,
    prompt,
    reference_image,
    block_level_scale_config,
    seed,
    num_inference_steps,
    save_dir,
    csv_path=None,
    sub_experiment="",
    extra_meta=None,
):
    """Run a single GA18C-style image generation with block-level IP-Adapter scale."""
    metadata = dict(extra_meta or {})
    save_dir = Path(save_dir)
    csv_path = Path(csv_path) if csv_path else None

    pipeline.set_ip_adapter_scale(block_level_scale_config)
    generator = make_generator(seed)

    with timer(f"GA18C block={sub_experiment} seed={seed}") as timing:
        result = pipeline(
            prompt=prompt,
            ip_adapter_image=reference_image,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]

    config_tag = metadata.get("config_tag", "default")
    default_name = f"ga18c_{sub_experiment}_{config_tag}_seed{seed}.png"
    image_name = _resolve_image_name(default_name, seed, metadata)
    image_path = save_image(result, save_dir / image_name, label=image_name)

    timestamp = now_iso()
    if csv_path is not None:
        row = [
            metadata.get("experiment", "exp3"),
            sub_experiment,
            "GA18C",
            seed,
            prompt,
            metadata.get("reference_image", "mamoeiro"),
            "",
            json.dumps(block_level_scale_config, sort_keys=True),
            metadata.get("ip_adapter_scale_active_value", ""),
            metadata.get("target_block", "block_0"),
            metadata.get("target_layers", ""),
            num_inference_steps,
            str(image_path.name),
            timing["elapsed_sec"],
            timestamp,
        ]
        append_csv_row(csv_path, row)

    return {
        "image": result,
        "image_path": image_path,
        "elapsed_sec": timing["elapsed_sec"],
        "timestamp": timestamp,
    }


def run_ga19b(
    asr_pipeline,
    audio_array,
    sampling_rate,
    ground_truth,
    model_name,
    sample_index,
    csv_path=None,
    sub_experiment="",
    extra_meta=None,
):
    """Run a single ASR inference on an audio sample and log WER/CER."""
    metadata = dict(extra_meta or {})
    csv_path = Path(csv_path) if csv_path else None

    with timer(f"GA19B model={model_name} sample={sample_index}") as timing:
        if isinstance(asr_pipeline, dict) and asr_pipeline.get("backend_type") == "whisper_generate":
            torch_module = _require_torch()
            processor = asr_pipeline["processor"]
            model = asr_pipeline["model"]
            audio_np = np.asarray(audio_array, dtype=np.float32)
            effective_sampling_rate = 16000 if sub_experiment == "5b" else (sampling_rate or 16000)
            model_inputs = processor(
                audio_np,
                sampling_rate=effective_sampling_rate,
                return_attention_mask=True,
                return_tensors="pt",
            )

            input_features = model_inputs["input_features"].to(model.device)
            model_dtype = getattr(model, "dtype", None)
            if model_dtype is not None:
                input_features = input_features.to(model_dtype)

            generate_kwargs = {"language": "en", "task": "transcribe"}
            attention_mask = model_inputs.get("attention_mask")
            if attention_mask is not None:
                generate_kwargs["attention_mask"] = attention_mask.to(model.device)

            with torch_module.no_grad():
                predicted_ids = model.generate(input_features, **generate_kwargs)
            predicted = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        else:
            result = asr_pipeline(audio_array)
            predicted = result["text"]

    wer_value, cer_value = compute_wer_cer(ground_truth, predicted)
    audio_duration = len(audio_array) / sampling_rate if sampling_rate else 0.0
    timestamp = now_iso()

    if csv_path is not None:
        row = [
            metadata.get("experiment", "exp5"),
            sub_experiment,
            "GA19B",
            model_name,
            sample_index,
            round(audio_duration, 3),
            sampling_rate,
            ground_truth,
            predicted,
            wer_value,
            cer_value,
            timing["elapsed_sec"],
            timestamp,
        ]
        append_csv_row(csv_path, row)

    return {
        "predicted": predicted,
        "wer": wer_value,
        "cer": cer_value,
        "time": timing["elapsed_sec"],
        "timestamp": timestamp,
    }


def run_ga19c(
    tts_pipeline,
    input_text,
    seed,
    save_dir,
    csv_path=None,
    sub_experiment="",
    extra_meta=None,
):
    """Run a single Bark TTS generation and log waveform statistics."""
    metadata = dict(extra_meta or {})
    save_dir = Path(save_dir)
    csv_path = Path(csv_path) if csv_path else None

    torch_module = _require_torch()
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)

    with timer(f"GA19C text='{input_text[:40]}...' seed={seed}") as timing:
        output = tts_pipeline(input_text)

    audio_array = output["audio"]
    sampling_rate = output["sampling_rate"]
    stats = get_audio_stats(audio_array)
    duration_sec = stats["duration_samples"] / sampling_rate if sampling_rate else 0.0

    text_tag = metadata.get("text_tag", "default")
    audio_name = f"ga19c_{sub_experiment}_{text_tag}_seed{seed}.wav"
    audio_path = save_audio(audio_array, sampling_rate, save_dir / audio_name, label=audio_name)

    timestamp = now_iso()
    row = [
        metadata.get("experiment", "exp6"),
        sub_experiment,
        "GA19C",
        seed,
        metadata.get("model_name", "suno/bark-small"),
        input_text,
        len(input_text.split()),
        metadata.get("text_category", "default"),
        str(audio_path.name),
        round(duration_sec, 3),
        sampling_rate,
        round(stats["rms"], 6),
        timing["elapsed_sec"],
        metadata.get("round_trip_transcription", ""),
        metadata.get("round_trip_wer", ""),
        timestamp,
    ]
    if csv_path is not None:
        append_csv_row(csv_path, row)
    return {
        "audio": audio_array,
        "sampling_rate": sampling_rate,
        "duration": duration_sec,
        "stats": stats,
        "time": timing["elapsed_sec"],
        "path": audio_path,
        "timestamp": timestamp,
    }