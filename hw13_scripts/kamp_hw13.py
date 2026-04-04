"""
HW13 Main Orchestrator — Multi-Modal Generative AI: IP-Adapter Image
Conditioning & Audio Generation/Understanding
Course: CS6078 Generative AI, University of Cincinnati, Spring 2026
Author: Ryan Kamp

This script coordinates all experimental runs for the HW13 project.
Experiments are grouped by model/modality to minimize loading overhead.
Supports checkpoint-based resumption for Colab session resilience.
"""

from __future__ import annotations

import csv
import json
import random
import sys
import traceback
from math import gcd
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - exercised only in missing-dependency environments
    torch = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "hw13_scripts"))

from hw13_data_utils import (
    EXPERIMENTS_DIR,
    PRINTOUTS_DIR,
    TeeLogger,
    append_csv_row,
    cleanup_pipeline,
    init_csv,
    load_cached_image,
    load_checkpoint,
    now_iso,
    save_checkpoint,
)
from hw13_experiment_runner import run_ga18b, run_ga18c, run_ga19b, run_ga19c


# --- Shared seed sets ---
SEEDS_3 = [42, 123, 456]
SEEDS_2 = [42, 123]
SEEDS_8 = [42, 123, 456, 789, 1024, 2048, 3000, 4096]


# --- Console capture ---
LOG_PATH = PRINTOUTS_DIR / "kamp_hw13_execution_log.txt"


# --- Vision model configuration ---
VISION_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_WEIGHTS = "ip-adapter_sdxl.bin"

GA18B_DEFAULT_PROMPT = ""
GA18B_DEFAULT_REFERENCE = "items_variation"
GA18B_DEFAULT_SCALE = 0.8
GA18B_DEFAULT_STEPS = 25

GA18C_DEFAULT_PROMPT = "a cat inside of a box"
GA18C_DEFAULT_REFERENCE = "mamoeiro"
GA18C_DEFAULT_BLOCK_CONFIG = {"up": {"block_0": [0.0, 1.0, 0.0]}}
GA18C_DEFAULT_STEPS = 25

GROUP_A_EXP1_HEADER = [
    "experiment",
    "script",
    "task_type",
    "seed",
    "prompt",
    "ip_adapter_scale",
    "num_inference_steps",
    "model",
    "reference_image",
    "input_text",
    "output_path",
    "generation_time_sec",
    "timestamp",
]

GROUP_A_EXP2_HEADER = [
    "experiment",
    "sub_experiment",
    "script",
    "seed",
    "prompt",
    "reference_image",
    "ip_adapter_scale",
    "block_level_scale_config",
    "num_inference_steps",
    "image_path",
    "generation_time_sec",
    "timestamp",
]

GROUP_A_EXP3_HEADER = [
    "experiment",
    "sub_experiment",
    "script",
    "seed",
    "prompt",
    "reference_image",
    "ip_adapter_scale",
    "block_level_scale_config",
    "ip_adapter_scale_active_value",
    "target_block",
    "target_layers",
    "num_inference_steps",
    "image_path",
    "generation_time_sec",
    "timestamp",
]

GROUP_A_EXP4_HEADER = [
    "experiment",
    "sub_experiment",
    "script",
    "seed",
    "prompt",
    "prompt_type",
    "reference_image",
    "ip_adapter_scale",
    "block_level_scale_config",
    "conditioning_mode",
    "num_inference_steps",
    "image_path",
    "generation_time_sec",
    "timestamp",
]

GROUP_A_EXP7_HEADER = [
    "experiment",
    "script",
    "paradigm",
    "seed",
    "prompt",
    "parameters_json",
    "output_path",
    "output_type",
    "generation_time_sec",
    "timestamp",
]

GROUP_B_DATASET_ID = "PolyAI/minds14"
GROUP_B_DATASET_CONFIG = "en-AU"

GROUP_B_MODEL_COMPARISON = [
    "facebook/wav2vec2-base-960h",
    "openai/whisper-tiny",
    "openai/whisper-small",
    "openai/whisper-medium",
]

GROUP_B_MISMATCH_MODELS = [
    ("default_pipeline", None),
    ("whisper_small", "openai/whisper-small"),
]

GROUP_B_MISMATCH_RATES = [8000, 16000, 22050, 48000]
GROUP_B_SAMPLE_SELECTION_SEED = 42

GROUP_B_EXP5_HEADER = [
    "experiment",
    "sub_experiment",
    "script",
    "model_name",
    "sample_index",
    "audio_duration_sec",
    "sampling_rate",
    "ground_truth_transcription",
    "predicted_transcription",
    "wer",
    "cer",
    "inference_time_sec",
    "timestamp",
]

GROUP_C_TTS_MODEL = "suno/bark-small"
GROUP_C_ROUND_TRIP_FALLBACK_ASR_MODEL = "openai/whisper-medium"

GA19C_DEFAULT_TEXT = (
    "Ladybugs have had important roles in culture and religion, being associated "
    "with luck, love, fertility and prophecy."
)

GROUP_C_EXP6A_SEEDS = [42, 123, 456, 789, 1024]

GROUP_C_EXP6B_TEXTS = [
    {
        "text_tag": "short",
        "text_category": "short",
        "input_text": "Hello, how are you?",
    },
    {
        "text_tag": "medium",
        "text_category": "medium",
        "input_text": GA19C_DEFAULT_TEXT,
    },
    {
        "text_tag": "long",
        "text_category": "long",
        "input_text": (
            "The quick brown fox jumps over the lazy dog. This sentence contains "
            "every letter in the English alphabet and has been used for typing "
            "practice since the late nineteenth century."
        ),
    },
]

GROUP_C_EXP6C_TEXTS = [
    {
        "text_tag": "conversational",
        "text_category": "conversational",
        "input_text": "Hey, could you grab me a coffee on your way back? Thanks a lot!",
    },
    {
        "text_tag": "technical",
        "text_category": "technical",
        "input_text": (
            "The convolutional neural network processes input tensors through "
            "sequential layers of learned filters."
        ),
    },
    {
        "text_tag": "proper_nouns",
        "text_category": "proper_nouns",
        "input_text": (
            "Professor Schwarzenegger from the University of Cincinnati presented "
            "the findings at the symposium."
        ),
    },
]

GROUP_C_EXP6D_TEXTS = [
    {
        "text_tag": "roundtrip_short",
        "text_category": "short",
        "input_text": GROUP_C_EXP6B_TEXTS[0]["input_text"],
    },
    {
        "text_tag": "roundtrip_technical",
        "text_category": "technical",
        "input_text": GROUP_C_EXP6C_TEXTS[1]["input_text"],
    },
    {
        "text_tag": "roundtrip_proper_nouns",
        "text_category": "proper_nouns",
        "input_text": GROUP_C_EXP6C_TEXTS[2]["input_text"],
    },
]

GROUP_C_EXP6_HEADER = [
    "experiment",
    "sub_experiment",
    "script",
    "seed",
    "model",
    "input_text",
    "text_length_words",
    "text_category",
    "output_path",
    "audio_duration_sec",
    "audio_sampling_rate",
    "waveform_rms",
    "generation_time_sec",
    "round_trip_transcription",
    "round_trip_wer",
    "timestamp",
]


def _cuda_available():
    """Return True when torch is installed and CUDA is available."""
    return bool(torch is not None and torch.cuda.is_available())


def _print_runtime_context():
    """Print basic environment context for experiment execution logs."""
    print(f"[ORCHESTRATOR] Started: {now_iso()}")
    print(f"[ORCHESTRATOR] torch installed: {torch is not None}")
    print(f"[ORCHESTRATOR] CUDA available: {_cuda_available()}")
    if _cuda_available():
        print(f"[ORCHESTRATOR] GPU: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[ORCHESTRATOR] VRAM: {total_memory:.1f} GB")


def _relative_output_path(path, experiments_root):
    """Return a stable experiment-relative output path string."""
    path = Path(path)
    experiments_root = Path(experiments_root)
    try:
        return str(path.relative_to(experiments_root))
    except ValueError:
        return str(path.name)


def _make_block_config(block_name, layer_values):
    """Construct the block-level IP-Adapter configuration dictionary."""
    return {"up": {block_name: list(layer_values)}}


def _prepare_group_a_layout(experiments_root):
    """Build the directory layout used by Group A runs."""
    experiments_root = Path(experiments_root)
    return {
        "root": experiments_root,
        "exp1_dir": experiments_root / "exp1_baselines",
        "exp2_dir": experiments_root / "exp2_ga18b_scale",
        "exp3_dir": experiments_root / "exp3_ga18c_blocks",
        "exp4_dir": experiments_root / "exp4_text_image_interaction",
        "exp7_dir": experiments_root / "exp7_seed_sensitivity",
    }


def _prepare_group_b_layout(experiments_root):
    """Build the directory layout used by Group B runs."""
    experiments_root = Path(experiments_root)
    return {
        "root": experiments_root,
        "exp1_dir": experiments_root / "exp1_baselines",
        "exp5_dir": experiments_root / "exp5_ga19b_asr",
    }


def _prepare_group_c_layout(experiments_root):
    """Build the directory layout used by Group C runs."""
    experiments_root = Path(experiments_root)
    return {
        "root": experiments_root,
        "exp1_dir": experiments_root / "exp1_baselines",
        "exp6_dir": experiments_root / "exp6_ga19c_tts",
        "exp7_dir": experiments_root / "exp7_seed_sensitivity",
    }


def _select_sample_indices(total_count, sample_count, seed=GROUP_B_SAMPLE_SELECTION_SEED):
    """Return a deterministic random sample of dataset indices."""
    if sample_count > total_count:
        raise ValueError(
            f"Requested {sample_count} samples from dataset of size {total_count}."
        )
    rng = random.Random(seed)
    return rng.sample(range(total_count), sample_count)


def _resolve_asr_model_name(asr_pipeline, fallback_name):
    """Resolve the loaded ASR model identifier for logging."""
    if isinstance(asr_pipeline, dict):
        return asr_pipeline.get("model_name") or fallback_name or "default_pipeline"

    model = getattr(asr_pipeline, "model", None)
    config = getattr(model, "config", None)
    resolved_name = getattr(config, "_name_or_path", None)
    return resolved_name or fallback_name or "default_pipeline"


def _select_best_group_b_asr_model(
    exp5_csv_path=None,
    comparison_sub_experiment="5a",
    fallback_model=GROUP_C_ROUND_TRIP_FALLBACK_ASR_MODEL,
):
    """Select the best Experiment 5 ASR model using average WER on the 5A subset."""
    csv_path = Path(exp5_csv_path or (EXPERIMENTS_DIR / "exp5_ga19b_asr" / "exp5_ga19b_asr.csv"))
    if not csv_path.exists():
        print(
            f"[GROUP C] Experiment 5 CSV missing at {csv_path}; "
            f"using fallback ASR model {fallback_model}."
        )
        return fallback_model

    aggregates = {}
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if comparison_sub_experiment and row.get("sub_experiment") != comparison_sub_experiment:
                continue

            model_name = (row.get("model_name") or "").strip()
            wer_text = row.get("wer", "")
            cer_text = row.get("cer", "")
            if not model_name or not wer_text or not cer_text:
                continue

            stats = aggregates.setdefault(
                model_name,
                {"count": 0, "wer_sum": 0.0, "cer_sum": 0.0},
            )
            stats["count"] += 1
            stats["wer_sum"] += float(wer_text)
            stats["cer_sum"] += float(cer_text)

    if not aggregates:
        print(
            f"[GROUP C] No usable rows found in {csv_path}; "
            f"using fallback ASR model {fallback_model}."
        )
        return fallback_model

    ranked_models = sorted(
        (
            stats["wer_sum"] / stats["count"],
            stats["cer_sum"] / stats["count"],
            -stats["count"],
            model_name,
        )
        for model_name, stats in aggregates.items()
    )
    best_wer, best_cer, _, best_model = ranked_models[0]
    print(
        "[GROUP C] Selected round-trip ASR model from Experiment 5: "
        f"{best_model} (avg WER={best_wer:.4f}, avg CER={best_cer:.4f}, "
        f"subset={comparison_sub_experiment})"
    )
    return best_model


def _resample_audio_for_asr(audio_array, source_rate, target_rate=16000):
    """Resample generated Bark audio to the ASR target rate when needed."""
    audio_np = np.asarray(audio_array, dtype=np.float32)
    audio_np = np.squeeze(audio_np)
    if not source_rate or int(source_rate) == int(target_rate):
        return audio_np

    try:
        from scipy.signal import resample_poly
    except ImportError as error:  # pragma: no cover - exercised only in missing-dependency environments
        raise ImportError("scipy is required for Group C round-trip resampling.") from error

    divisor = gcd(int(source_rate), int(target_rate))
    up = int(target_rate) // divisor
    down = int(source_rate) // divisor
    return resample_poly(audio_np, up, down).astype(np.float32)


def load_group_b_datasets():
    """Load the MINDS-14 dataset and the resampled variants used by Group B."""
    from datasets import Audio, load_dataset

    minds_raw = load_dataset(GROUP_B_DATASET_ID, name=GROUP_B_DATASET_CONFIG, split="train")
    datasets_by_rate = {
        8000: minds_raw,
        16000: minds_raw.cast_column("audio", Audio(sampling_rate=16000)),
        22050: minds_raw.cast_column("audio", Audio(sampling_rate=22050)),
        48000: minds_raw.cast_column("audio", Audio(sampling_rate=48000)),
    }
    return minds_raw, datasets_by_rate


def load_group_b_asr_pipeline(model_name=None, device=None):
    """Load a single ASR pipeline for Group B on the active device."""
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    resolved_device = device if device is not None else (0 if _cuda_available() else -1)

    if model_name is not None and "whisper" in model_name.lower():
        if torch is None:
            raise ImportError("torch is required for Whisper ASR backends.")

        torch_dtype = torch.float16 if _cuda_available() else torch.float32
        model_kwargs = {
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
        }

        try:
            whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                dtype=torch_dtype,
                **model_kwargs,
            )
        except TypeError:
            whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                **model_kwargs,
            )

        device_name = "cuda" if _cuda_available() else "cpu"
        whisper_model = whisper_model.to(device_name)
        whisper_processor = AutoProcessor.from_pretrained(model_name)
        whisper_backend = {
            "backend_type": "whisper_generate",
            "model_name": model_name,
            "model": whisper_model,
            "processor": whisper_processor,
            "device": device_name,
        }
        return whisper_backend, model_name, resolved_device

    pipeline_kwargs = {"device": resolved_device}
    if model_name is not None:
        pipeline_kwargs["model"] = model_name

    asr_pipeline = pipeline("automatic-speech-recognition", **pipeline_kwargs)
    resolved_name = _resolve_asr_model_name(asr_pipeline, model_name)
    return asr_pipeline, resolved_name, resolved_device


def load_group_c_tts_pipeline(device=None):
    """Load the Bark text-to-speech pipeline used by Group C."""
    from transformers import pipeline

    resolved_device = device if device is not None else (0 if _cuda_available() else -1)
    tts_pipeline = pipeline(
        "text-to-speech",
        model=GROUP_C_TTS_MODEL,
        device=resolved_device,
    )
    return tts_pipeline, GROUP_C_TTS_MODEL, resolved_device


def load_group_a_pipeline(device=None):
    """Load the SDXL Base + IP-Adapter pipeline used by the Group A notebooks."""
    if torch is None:
        raise ImportError("torch is required to load the Group A SDXL pipeline.")

    from diffusers import StableDiffusionXLPipeline

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pipeline_kwargs = {"torch_dtype": torch.float16, "variant": "fp16"}
    if resolved_device != "cuda":
        pipeline_kwargs = {"torch_dtype": torch.float32}

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        VISION_MODEL_ID,
        **pipeline_kwargs,
    )
    pipeline = pipeline.to(resolved_device)
    pipeline.load_ip_adapter(
        IP_ADAPTER_REPO,
        subfolder=IP_ADAPTER_SUBFOLDER,
        weight_name=IP_ADAPTER_WEIGHTS,
    )
    return pipeline, resolved_device


def main():
    """Main entry point. Runs all execution groups with checkpointing."""
    tee = TeeLogger(LOG_PATH).start()
    _print_runtime_context()

    checkpoint = load_checkpoint()
    completed_groups = set()
    if checkpoint:
        completed_groups = set(checkpoint.get("completed_groups", []))
        print(f"[ORCHESTRATOR] Resuming. Completed groups: {sorted(completed_groups)}")

    execution_groups = [
        ("A", run_group_a),
        ("B", run_group_b),
        ("C", run_group_c),
    ]

    try:
        for group_name, group_func in execution_groups:
            if group_name in completed_groups:
                print(f"[ORCHESTRATOR] Skipping Group {group_name} (already completed)")
                continue

            group_func()
            completed_groups.add(group_name)
            save_checkpoint(
                {
                    "completed_groups": sorted(completed_groups),
                    "last_completed_step": f"Group {group_name}",
                }
            )

        print(f"\n[ORCHESTRATOR] All experiments complete: {now_iso()}")

    except Exception as error:
        print(f"\n[ORCHESTRATOR] ERROR: {error}")
        traceback.print_exc()
        save_checkpoint(
            {
                "completed_groups": sorted(completed_groups),
                "error": str(error),
            }
        )
        raise
    finally:
        tee.stop()


def run_group_a(
    dry_run=False,
    pipeline=None,
    experiments_dir=None,
    checkpoint_enabled=True,
    resume=False,
):
    """
    Group A: SDXL Base 1.0 + IP-Adapter — All Vision Experiments.
    Exp 1 (GA18B + GA18C baselines), Exp 2 (full), Exp 3 (full),
    Exp 4 (full), Exp 7 (GA18B + GA18C seeds).
    """
    return _run_group_a_impl(
        dry_run=dry_run,
        pipeline=pipeline,
        experiments_dir=experiments_dir,
        checkpoint_enabled=checkpoint_enabled,
        resume=resume,
    )


def _run_group_a_impl(
    dry_run=False,
    pipeline=None,
    experiments_dir=None,
    checkpoint_enabled=True,
    resume=False,
):
    """Execute Group A in either full or dry-run mode."""
    layout = _prepare_group_a_layout(experiments_dir or EXPERIMENTS_DIR)
    experiments_root = layout["root"]

    print(f"\n{'=' * 60}")
    print(f"[GROUP A] Starting {'dry-run' if dry_run else 'full'} vision workflow")
    print(f"[GROUP A] Output root: {experiments_root}")

    exp1_csv = init_csv(layout["exp1_dir"] / "exp1_baselines.csv", GROUP_A_EXP1_HEADER)
    exp2_csv = init_csv(layout["exp2_dir"] / "exp2_ga18b_scale.csv", GROUP_A_EXP2_HEADER)
    exp3_csv = init_csv(layout["exp3_dir"] / "exp3_ga18c_blocks.csv", GROUP_A_EXP3_HEADER)
    exp4_csv = init_csv(
        layout["exp4_dir"] / "exp4_text_image_interaction.csv",
        GROUP_A_EXP4_HEADER,
    )
    exp7_csv = init_csv(layout["exp7_dir"] / "exp7_seed_sensitivity.csv", GROUP_A_EXP7_HEADER)

    items_reference = load_cached_image(GA18B_DEFAULT_REFERENCE).resize((1024, 1024))
    mamoeiro_reference = load_cached_image(GA18C_DEFAULT_REFERENCE).resize((1024, 1024))

    pipeline_loaded_here = pipeline is None
    if pipeline_loaded_here:
        pipeline, resolved_device = load_group_a_pipeline()
        print(f"[GROUP A] Model loaded on {resolved_device}.")
    else:
        print("[GROUP A] Using caller-provided pipeline.")

    summary = {
        "mode": "dry-run" if dry_run else "full",
        "output_root": str(experiments_root),
        "total_images": 0,
        "per_experiment": {"exp1": 0, "exp2": 0, "exp3": 0, "exp4": 0, "exp7": 0},
    }

    completed_sub_experiments = set()
    completed_groups = set()
    if checkpoint_enabled and resume and not dry_run:
        checkpoint = load_checkpoint()
        if checkpoint:
            completed_sub_experiments = set(checkpoint.get("group_a_completed_sub_experiments", []))
            completed_groups = set(checkpoint.get("completed_groups", []))
            print(
                "[GROUP A] Resuming from checkpoint. "
                f"Completed sub-experiments: {sorted(completed_sub_experiments)}"
            )

    def record_run(experiment_key):
        summary["total_images"] += 1
        summary["per_experiment"][experiment_key] += 1

    def mark_sub_experiment_complete(label):
        if not checkpoint_enabled or dry_run:
            return
        completed_sub_experiments.add(label)
        save_checkpoint(
            {
                "completed_groups": sorted(completed_groups),
                "group_a_completed_sub_experiments": sorted(completed_sub_experiments),
                "last_completed_step": f"Group A {label}",
            }
        )

    def should_skip(label):
        return resume and checkpoint_enabled and not dry_run and label in completed_sub_experiments

    def log_exp1(script_name, task_type, seed, prompt, scale_value, steps, reference_name, result_meta):
        append_csv_row(
            exp1_csv,
            [
                "exp1",
                script_name,
                task_type,
                seed,
                prompt,
                scale_value,
                steps,
                f"{VISION_MODEL_ID} + {IP_ADAPTER_REPO}",
                reference_name,
                "",
                _relative_output_path(result_meta["image_path"], experiments_root),
                result_meta["elapsed_sec"],
                result_meta["timestamp"],
            ],
        )

    def log_exp2(sub_experiment, seed, prompt, reference_name, scale_value, steps, result_meta):
        append_csv_row(
            exp2_csv,
            [
                "exp2",
                sub_experiment,
                "GA18B",
                seed,
                prompt,
                reference_name,
                scale_value,
                "",
                steps,
                _relative_output_path(result_meta["image_path"], experiments_root),
                result_meta["elapsed_sec"],
                result_meta["timestamp"],
            ],
        )

    def log_exp3(
        sub_experiment,
        seed,
        prompt,
        reference_name,
        block_config,
        active_value,
        target_block,
        target_layers,
        steps,
        result_meta,
    ):
        append_csv_row(
            exp3_csv,
            [
                "exp3",
                sub_experiment,
                "GA18C",
                seed,
                prompt,
                reference_name,
                "",
                json.dumps(block_config, sort_keys=True),
                active_value,
                target_block,
                target_layers,
                steps,
                _relative_output_path(result_meta["image_path"], experiments_root),
                result_meta["elapsed_sec"],
                result_meta["timestamp"],
            ],
        )

    def log_exp4(
        sub_experiment,
        script_name,
        seed,
        prompt,
        prompt_type,
        reference_name,
        scale_value,
        block_config,
        conditioning_mode,
        steps,
        result_meta,
    ):
        append_csv_row(
            exp4_csv,
            [
                "exp4",
                sub_experiment,
                script_name,
                seed,
                prompt,
                prompt_type,
                reference_name,
                scale_value,
                json.dumps(block_config, sort_keys=True) if block_config else "",
                conditioning_mode,
                steps,
                _relative_output_path(result_meta["image_path"], experiments_root),
                result_meta["elapsed_sec"],
                result_meta["timestamp"],
            ],
        )

    def log_exp7(script_name, paradigm, seed, prompt, parameters_json, result_meta):
        append_csv_row(
            exp7_csv,
            [
                "exp7",
                script_name,
                paradigm,
                seed,
                prompt,
                json.dumps(parameters_json, sort_keys=True),
                _relative_output_path(result_meta["image_path"], experiments_root),
                "image",
                result_meta["elapsed_sec"],
                result_meta["timestamp"],
            ],
        )

    try:
        if should_skip("exp1"):
            print("[GROUP A] Skipping Exp 1 (already completed)")
        else:
            print("[GROUP A] Exp 1: Baselines")
            ga18b_baseline = run_ga18b(
                pipeline,
                prompt=GA18B_DEFAULT_PROMPT,
                reference_image=items_reference,
                ip_adapter_scale=GA18B_DEFAULT_SCALE,
                seed=42,
                num_inference_steps=GA18B_DEFAULT_STEPS,
                save_dir=layout["exp1_dir"],
                csv_path=None,
                sub_experiment="baseline",
                extra_meta={"filename_stem": "ga18b_baseline_default"},
            )
            log_exp1(
                "GA18B",
                "image_variation",
                42,
                GA18B_DEFAULT_PROMPT,
                GA18B_DEFAULT_SCALE,
                GA18B_DEFAULT_STEPS,
                GA18B_DEFAULT_REFERENCE,
                ga18b_baseline,
            )
            record_run("exp1")

            ga18c_baseline = run_ga18c(
                pipeline,
                prompt=GA18C_DEFAULT_PROMPT,
                reference_image=mamoeiro_reference,
                block_level_scale_config=GA18C_DEFAULT_BLOCK_CONFIG,
                seed=42,
                num_inference_steps=GA18C_DEFAULT_STEPS,
                save_dir=layout["exp1_dir"],
                csv_path=None,
                sub_experiment="baseline",
                extra_meta={"filename_stem": "ga18c_baseline_default"},
            )
            log_exp1(
                "GA18C",
                "style_transfer",
                42,
                GA18C_DEFAULT_PROMPT,
                json.dumps(GA18C_DEFAULT_BLOCK_CONFIG, sort_keys=True),
                GA18C_DEFAULT_STEPS,
                GA18C_DEFAULT_REFERENCE,
                ga18c_baseline,
            )
            record_run("exp1")
            mark_sub_experiment_complete("exp1")

        if should_skip("2a"):
            print("[GROUP A] Skipping Exp 2A (already completed)")
        else:
            print("[GROUP A] Exp 2A: Uniform scale sweep")
            scale_values = [0.5] if dry_run else [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            scale_seeds = [42] if dry_run else SEEDS_3
            for scale_value in scale_values:
                for seed in scale_seeds:
                    result = run_ga18b(
                        pipeline,
                        prompt=GA18B_DEFAULT_PROMPT,
                        reference_image=items_reference,
                        ip_adapter_scale=scale_value,
                        seed=seed,
                        num_inference_steps=GA18B_DEFAULT_STEPS,
                        save_dir=layout["exp2_dir"],
                        csv_path=None,
                        sub_experiment="2a",
                        extra_meta={
                            "filename_stem": f"ga18b_2a_scale_{scale_value:.2f}",
                        },
                    )
                    log_exp2("2a", seed, GA18B_DEFAULT_PROMPT, GA18B_DEFAULT_REFERENCE, scale_value, GA18B_DEFAULT_STEPS, result)
                    record_run("exp2")
            mark_sub_experiment_complete("2a")

        if not dry_run:
            if should_skip("2b"):
                print("[GROUP A] Skipping Exp 2B (already completed)")
            else:
                print("[GROUP A] Exp 2B: Alternative reference sweep")
                for scale_value in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                    for seed in SEEDS_2:
                        result = run_ga18b(
                            pipeline,
                            prompt=GA18B_DEFAULT_PROMPT,
                            reference_image=mamoeiro_reference,
                            ip_adapter_scale=scale_value,
                            seed=seed,
                            num_inference_steps=GA18B_DEFAULT_STEPS,
                            save_dir=layout["exp2_dir"],
                            csv_path=None,
                            sub_experiment="2b",
                            extra_meta={
                                "filename_stem": f"ga18b_2b_scale_{scale_value:.2f}",
                            },
                        )
                        log_exp2("2b", seed, GA18B_DEFAULT_PROMPT, GA18C_DEFAULT_REFERENCE, scale_value, GA18B_DEFAULT_STEPS, result)
                        record_run("exp2")
                mark_sub_experiment_complete("2b")

            if should_skip("2c"):
                print("[GROUP A] Skipping Exp 2C (already completed)")
            else:
                print("[GROUP A] Exp 2C: Inference-step sweep")
                for step_count in [10, 25, 50]:
                    for seed in SEEDS_2:
                        result = run_ga18b(
                            pipeline,
                            prompt=GA18B_DEFAULT_PROMPT,
                            reference_image=items_reference,
                            ip_adapter_scale=GA18B_DEFAULT_SCALE,
                            seed=seed,
                            num_inference_steps=step_count,
                            save_dir=layout["exp2_dir"],
                            csv_path=None,
                            sub_experiment="2c",
                            extra_meta={
                                "filename_stem": f"ga18b_2c_steps_{step_count}",
                            },
                        )
                        log_exp2("2c", seed, GA18B_DEFAULT_PROMPT, GA18B_DEFAULT_REFERENCE, GA18B_DEFAULT_SCALE, step_count, result)
                        record_run("exp2")
                mark_sub_experiment_complete("2c")

        if should_skip("3a"):
            print("[GROUP A] Skipping Exp 3A (already completed)")
        else:
            print("[GROUP A] Exp 3A: Layer variation within block_0")
            layer_configs = [([0.0, 1.0, 0.0], "010")] if dry_run else [
                ([1.0, 0.0, 0.0], "100"),
                ([0.0, 1.0, 0.0], "010"),
                ([0.0, 0.0, 1.0], "001"),
                ([1.0, 1.0, 0.0], "110"),
                ([0.0, 1.0, 1.0], "011"),
                ([1.0, 1.0, 1.0], "111"),
            ]
            layer_seeds = [42] if dry_run else SEEDS_3
            for layer_values, layer_tag in layer_configs:
                block_config = _make_block_config("block_0", layer_values)
                for seed in layer_seeds:
                    result = run_ga18c(
                        pipeline,
                        prompt=GA18C_DEFAULT_PROMPT,
                        reference_image=mamoeiro_reference,
                        block_level_scale_config=block_config,
                        seed=seed,
                        num_inference_steps=GA18C_DEFAULT_STEPS,
                        save_dir=layout["exp3_dir"],
                        csv_path=None,
                        sub_experiment="3a",
                        extra_meta={
                            "filename_stem": f"ga18c_3a_layer_{layer_tag}",
                            "config_tag": f"layer_{layer_tag}",
                        },
                    )
                    log_exp3(
                        "3a",
                        seed,
                        GA18C_DEFAULT_PROMPT,
                        GA18C_DEFAULT_REFERENCE,
                        block_config,
                        max(layer_values),
                        "block_0",
                        layer_tag,
                        GA18C_DEFAULT_STEPS,
                        result,
                    )
                    record_run("exp3")
            mark_sub_experiment_complete("3a")

        if not dry_run:
            if should_skip("3b"):
                print("[GROUP A] Skipping Exp 3B (already completed)")
            else:
                print("[GROUP A] Exp 3B: Block comparison")
                for block_name, block_tag in [("block_0", "block0"), ("block_1", "block1"), ("block_2", "block2")]:
                    block_config = _make_block_config(block_name, [0.0, 1.0, 0.0])
                    for seed in SEEDS_2:
                        result = run_ga18c(
                            pipeline,
                            prompt=GA18C_DEFAULT_PROMPT,
                            reference_image=mamoeiro_reference,
                            block_level_scale_config=block_config,
                            seed=seed,
                            num_inference_steps=GA18C_DEFAULT_STEPS,
                            save_dir=layout["exp3_dir"],
                            csv_path=None,
                            sub_experiment="3b",
                            extra_meta={
                                "filename_stem": f"ga18c_3b_{block_tag}",
                                "config_tag": block_tag,
                            },
                        )
                        log_exp3(
                            "3b",
                            seed,
                            GA18C_DEFAULT_PROMPT,
                            GA18C_DEFAULT_REFERENCE,
                            block_config,
                            1.0,
                            block_name,
                            "010",
                            GA18C_DEFAULT_STEPS,
                            result,
                        )
                        record_run("exp3")
                mark_sub_experiment_complete("3b")

            if should_skip("3c"):
                print("[GROUP A] Skipping Exp 3C (already completed)")
            else:
                print("[GROUP A] Exp 3C: Scale-intensity sweep")
                for scale_value in [0.25, 0.5, 0.75, 1.0, 1.5]:
                    block_config = _make_block_config("block_0", [0.0, scale_value, 0.0])
                    for seed in SEEDS_2:
                        result = run_ga18c(
                            pipeline,
                            prompt=GA18C_DEFAULT_PROMPT,
                            reference_image=mamoeiro_reference,
                            block_level_scale_config=block_config,
                            seed=seed,
                            num_inference_steps=GA18C_DEFAULT_STEPS,
                            save_dir=layout["exp3_dir"],
                            csv_path=None,
                            sub_experiment="3c",
                            extra_meta={
                                "filename_stem": f"ga18c_3c_scale_{scale_value:.2f}",
                                "config_tag": f"scale_{scale_value:.2f}",
                            },
                        )
                        log_exp3(
                            "3c",
                            seed,
                            GA18C_DEFAULT_PROMPT,
                            GA18C_DEFAULT_REFERENCE,
                            block_config,
                            scale_value,
                            "block_0",
                            "010",
                            GA18C_DEFAULT_STEPS,
                            result,
                        )
                        record_run("exp3")
                mark_sub_experiment_complete("3c")

            if should_skip("3d"):
                print("[GROUP A] Skipping Exp 3D (already completed)")
            else:
                print("[GROUP A] Exp 3D: Alternative prompts")
                prompt_map = [
                    ("cat", "a cat inside of a box"),
                    ("mountain", "a mountain landscape at sunset"),
                    ("portrait", "a portrait of an old man"),
                    ("cityscape", "a futuristic cityscape"),
                ]
                for prompt_tag, prompt_text in prompt_map:
                    for seed in SEEDS_2:
                        result = run_ga18c(
                            pipeline,
                            prompt=prompt_text,
                            reference_image=mamoeiro_reference,
                            block_level_scale_config=GA18C_DEFAULT_BLOCK_CONFIG,
                            seed=seed,
                            num_inference_steps=GA18C_DEFAULT_STEPS,
                            save_dir=layout["exp3_dir"],
                            csv_path=None,
                            sub_experiment="3d",
                            extra_meta={
                                "filename_stem": f"ga18c_3d_prompt_{prompt_tag}",
                                "config_tag": f"prompt_{prompt_tag}",
                            },
                        )
                        log_exp3(
                            "3d",
                            seed,
                            prompt_text,
                            GA18C_DEFAULT_REFERENCE,
                            GA18C_DEFAULT_BLOCK_CONFIG,
                            1.0,
                            "block_0",
                            "010",
                            GA18C_DEFAULT_STEPS,
                            result,
                        )
                        record_run("exp3")
                mark_sub_experiment_complete("3d")

            if should_skip("3e"):
                print("[GROUP A] Skipping Exp 3E (already completed)")
            else:
                print("[GROUP A] Exp 3E: Alternative style reference")
                reference_map = [
                    (GA18C_DEFAULT_REFERENCE, mamoeiro_reference),
                    (GA18B_DEFAULT_REFERENCE, items_reference),
                ]
                for reference_name, reference_image in reference_map:
                    for seed in SEEDS_2:
                        result = run_ga18c(
                            pipeline,
                            prompt=GA18C_DEFAULT_PROMPT,
                            reference_image=reference_image,
                            block_level_scale_config=GA18C_DEFAULT_BLOCK_CONFIG,
                            seed=seed,
                            num_inference_steps=GA18C_DEFAULT_STEPS,
                            save_dir=layout["exp3_dir"],
                            csv_path=None,
                            sub_experiment="3e",
                            extra_meta={
                                "filename_stem": f"ga18c_3e_ref_{reference_name}",
                                "config_tag": f"ref_{reference_name}",
                            },
                        )
                        log_exp3(
                            "3e",
                            seed,
                            GA18C_DEFAULT_PROMPT,
                            reference_name,
                            GA18C_DEFAULT_BLOCK_CONFIG,
                            1.0,
                            "block_0",
                            "010",
                            GA18C_DEFAULT_STEPS,
                            result,
                        )
                        record_run("exp3")
                mark_sub_experiment_complete("3e")

        if should_skip("4a"):
            print("[GROUP A] Skipping Exp 4A (already completed)")
        else:
            print("[GROUP A] Exp 4A: Uniform scale with active text")
            interaction_prompt = "a beautiful sunset over the ocean"
            scale_values = [0.4] if dry_run else [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            interaction_seeds = [42] if dry_run else SEEDS_2
            for scale_value in scale_values:
                for seed in interaction_seeds:
                    result = run_ga18b(
                        pipeline,
                        prompt=interaction_prompt,
                        reference_image=items_reference,
                        ip_adapter_scale=scale_value,
                        seed=seed,
                        num_inference_steps=GA18B_DEFAULT_STEPS,
                        save_dir=layout["exp4_dir"],
                        csv_path=None,
                        sub_experiment="4a",
                        extra_meta={
                            "filename_stem": f"ga18b_4a_scale_{scale_value:.2f}",
                        },
                    )
                    log_exp4(
                        "4a",
                        "GA18B",
                        seed,
                        interaction_prompt,
                        "moderate",
                        GA18B_DEFAULT_REFERENCE,
                        scale_value,
                        None,
                        "uniform",
                        GA18B_DEFAULT_STEPS,
                        result,
                    )
                    record_run("exp4")
            mark_sub_experiment_complete("4a")

        if not dry_run:
            if should_skip("4b"):
                print("[GROUP A] Skipping Exp 4B (already completed)")
            else:
                print("[GROUP A] Exp 4B: Prompt specificity")
                prompt_map = [
                    ("minimal", "scene"),
                    ("moderate", "a beautiful sunset over the ocean"),
                    (
                        "detailed",
                        "a photorealistic panoramic view of a dramatic sunset over a calm tropical ocean, vivid orange and purple sky, detailed 8k",
                    ),
                ]
                for prompt_type, prompt_text in prompt_map:
                    for seed in SEEDS_2:
                        result = run_ga18b(
                            pipeline,
                            prompt=prompt_text,
                            reference_image=items_reference,
                            ip_adapter_scale=0.5,
                            seed=seed,
                            num_inference_steps=GA18B_DEFAULT_STEPS,
                            save_dir=layout["exp4_dir"],
                            csv_path=None,
                            sub_experiment="4b",
                            extra_meta={
                                "filename_stem": f"ga18b_4b_prompt_{prompt_type}",
                            },
                        )
                        log_exp4(
                            "4b",
                            "GA18B",
                            seed,
                            prompt_text,
                            prompt_type,
                            GA18B_DEFAULT_REFERENCE,
                            0.5,
                            None,
                            "uniform",
                            GA18B_DEFAULT_STEPS,
                            result,
                        )
                        record_run("exp4")
                mark_sub_experiment_complete("4b")

            if should_skip("4c"):
                print("[GROUP A] Skipping Exp 4C (already completed)")
            else:
                print("[GROUP A] Exp 4C: Semantic compatibility")
                compatibility_prompts = [
                    ("compatible", "household items on a table"),
                    ("incompatible", "a mountain landscape at sunset"),
                    ("neutral", "an abstract painting"),
                ]
                for prompt_type, prompt_text in compatibility_prompts:
                    for seed in SEEDS_2:
                        result = run_ga18b(
                            pipeline,
                            prompt=prompt_text,
                            reference_image=items_reference,
                            ip_adapter_scale=0.5,
                            seed=seed,
                            num_inference_steps=GA18B_DEFAULT_STEPS,
                            save_dir=layout["exp4_dir"],
                            csv_path=None,
                            sub_experiment="4c",
                            extra_meta={
                                "filename_stem": f"ga18b_4c_prompt_{prompt_type}",
                            },
                        )
                        log_exp4(
                            "4c",
                            "GA18B",
                            seed,
                            prompt_text,
                            prompt_type,
                            GA18B_DEFAULT_REFERENCE,
                            0.5,
                            None,
                            "uniform",
                            GA18B_DEFAULT_STEPS,
                            result,
                        )
                        record_run("exp4")
                mark_sub_experiment_complete("4c")

            if should_skip("4d"):
                print("[GROUP A] Skipping Exp 4D (already completed)")
            else:
                print("[GROUP A] Exp 4D: Block-level prompt comparison")
                prompt_map = [
                    ("cat", "a cat inside of a box"),
                    ("sunset", "a beautiful sunset over the ocean"),
                ]
                for prompt_type, prompt_text in prompt_map:
                    for seed in SEEDS_2:
                        result = run_ga18c(
                            pipeline,
                            prompt=prompt_text,
                            reference_image=mamoeiro_reference,
                            block_level_scale_config=GA18C_DEFAULT_BLOCK_CONFIG,
                            seed=seed,
                            num_inference_steps=GA18C_DEFAULT_STEPS,
                            save_dir=layout["exp4_dir"],
                            csv_path=None,
                            sub_experiment="4d",
                            extra_meta={
                                "filename_stem": f"ga18c_4d_prompt_{prompt_type}",
                                "config_tag": f"prompt_{prompt_type}",
                            },
                        )
                        log_exp4(
                            "4d",
                            "GA18C",
                            seed,
                            prompt_text,
                            prompt_type,
                            GA18C_DEFAULT_REFERENCE,
                            "",
                            GA18C_DEFAULT_BLOCK_CONFIG,
                            "block",
                            GA18C_DEFAULT_STEPS,
                            result,
                        )
                        record_run("exp4")
                mark_sub_experiment_complete("4d")

            if should_skip("4e"):
                print("[GROUP A] Skipping Exp 4E (already completed)")
            else:
                print("[GROUP A] Exp 4E: Uniform vs. block-level comparison")
                for seed in SEEDS_3:
                    uniform_result = run_ga18b(
                        pipeline,
                        prompt=GA18C_DEFAULT_PROMPT,
                        reference_image=mamoeiro_reference,
                        ip_adapter_scale=0.5,
                        seed=seed,
                        num_inference_steps=GA18B_DEFAULT_STEPS,
                        save_dir=layout["exp4_dir"],
                        csv_path=None,
                        sub_experiment="4e",
                        extra_meta={"filename_stem": "ga18b_4e_mode_uniform"},
                    )
                    log_exp4(
                        "4e",
                        "GA18B",
                        seed,
                        GA18C_DEFAULT_PROMPT,
                        "matched_content",
                        GA18C_DEFAULT_REFERENCE,
                        0.5,
                        None,
                        "uniform",
                        GA18B_DEFAULT_STEPS,
                        uniform_result,
                    )
                    record_run("exp4")

                    block_result = run_ga18c(
                        pipeline,
                        prompt=GA18C_DEFAULT_PROMPT,
                        reference_image=mamoeiro_reference,
                        block_level_scale_config=GA18C_DEFAULT_BLOCK_CONFIG,
                        seed=seed,
                        num_inference_steps=GA18C_DEFAULT_STEPS,
                        save_dir=layout["exp4_dir"],
                        csv_path=None,
                        sub_experiment="4e",
                        extra_meta={
                            "filename_stem": "ga18c_4e_mode_block",
                            "config_tag": "mode_block",
                        },
                    )
                    log_exp4(
                        "4e",
                        "GA18C",
                        seed,
                        GA18C_DEFAULT_PROMPT,
                        "matched_content",
                        GA18C_DEFAULT_REFERENCE,
                        "",
                        GA18C_DEFAULT_BLOCK_CONFIG,
                        "block",
                        GA18C_DEFAULT_STEPS,
                        block_result,
                    )
                    record_run("exp4")
                mark_sub_experiment_complete("4e")

        if should_skip("exp7"):
            print("[GROUP A] Skipping Exp 7 (already completed)")
        else:
            print("[GROUP A] Exp 7: Seed sensitivity")
            seed_values = [42] if dry_run else SEEDS_8
            for seed in seed_values:
                ga18b_seed_result = run_ga18b(
                    pipeline,
                    prompt=GA18B_DEFAULT_PROMPT,
                    reference_image=items_reference,
                    ip_adapter_scale=GA18B_DEFAULT_SCALE,
                    seed=seed,
                    num_inference_steps=GA18B_DEFAULT_STEPS,
                    save_dir=layout["exp7_dir"],
                    csv_path=None,
                    sub_experiment="seed",
                    extra_meta={"filename_stem": "ga18b_seed"},
                )
                log_exp7(
                    "GA18B",
                    "vision_uniform",
                    seed,
                    GA18B_DEFAULT_PROMPT,
                    {
                        "ip_adapter_scale": GA18B_DEFAULT_SCALE,
                        "num_inference_steps": GA18B_DEFAULT_STEPS,
                        "reference_image": GA18B_DEFAULT_REFERENCE,
                    },
                    ga18b_seed_result,
                )
                record_run("exp7")

                ga18c_seed_result = run_ga18c(
                    pipeline,
                    prompt=GA18C_DEFAULT_PROMPT,
                    reference_image=mamoeiro_reference,
                    block_level_scale_config=GA18C_DEFAULT_BLOCK_CONFIG,
                    seed=seed,
                    num_inference_steps=GA18C_DEFAULT_STEPS,
                    save_dir=layout["exp7_dir"],
                    csv_path=None,
                    sub_experiment="seed",
                    extra_meta={
                        "filename_stem": "ga18c_seed",
                        "config_tag": "seed",
                    },
                )
                log_exp7(
                    "GA18C",
                    "vision_block",
                    seed,
                    GA18C_DEFAULT_PROMPT,
                    {
                        "block_level_scale_config": GA18C_DEFAULT_BLOCK_CONFIG,
                        "num_inference_steps": GA18C_DEFAULT_STEPS,
                        "reference_image": GA18C_DEFAULT_REFERENCE,
                    },
                    ga18c_seed_result,
                )
                record_run("exp7")
            mark_sub_experiment_complete("exp7")

        if checkpoint_enabled and not dry_run:
            completed_groups.add("A")
            save_checkpoint(
                {
                    "completed_groups": sorted(completed_groups),
                    "group_a_completed_sub_experiments": sorted(completed_sub_experiments),
                    "last_completed_step": "Group A complete",
                }
            )

        print(
            "[GROUP A] Complete. "
            f"Generated {summary['total_images']} images "
            f"({summary['per_experiment']})."
        )
        return summary
    finally:
        if pipeline_loaded_here:
            cleanup_pipeline(pipeline, "SDXL Base 1.0 + IP-Adapter")


def run_group_b(
    dry_run=False,
    experiments_dir=None,
    checkpoint_enabled=True,
    resume=False,
):
    """
    Group B: ASR Models — Audio Recognition Experiments.
    Exp 1 (GA19A + GA19B baselines), Exp 5 (full).
    """
    layout = _prepare_group_b_layout(experiments_dir or EXPERIMENTS_DIR)
    experiments_root = layout["root"]

    print(f"\n{'=' * 60}")
    print(f"[GROUP B] Starting {'dry-run' if dry_run else 'full'} ASR workflow")
    print(f"[GROUP B] Output root: {experiments_root}")

    exp1_csv = init_csv(layout["exp1_dir"] / "exp1_baselines.csv", GROUP_A_EXP1_HEADER)
    exp5_csv = init_csv(layout["exp5_dir"] / "exp5_ga19b_asr.csv", GROUP_B_EXP5_HEADER)

    summary = {
        "mode": "dry-run" if dry_run else "full",
        "output_root": str(experiments_root),
        "dataset_inspections": 0,
        "total_transcriptions": 0,
        "per_experiment": {"exp1": 0, "exp5": 0},
        "models_used": [],
    }

    completed_sub_experiments = set()
    completed_groups = set()
    if checkpoint_enabled and resume and not dry_run:
        checkpoint = load_checkpoint()
        if checkpoint:
            completed_sub_experiments = set(checkpoint.get("group_b_completed_sub_experiments", []))
            completed_groups = set(checkpoint.get("completed_groups", []))
            print(
                "[GROUP B] Resuming from checkpoint. "
                f"Completed labels: {sorted(completed_sub_experiments)}"
            )

    def record_transcription(experiment_key, model_name):
        summary["total_transcriptions"] += 1
        summary["per_experiment"][experiment_key] += 1
        if model_name not in summary["models_used"]:
            summary["models_used"].append(model_name)

    def mark_sub_experiment_complete(label):
        if not checkpoint_enabled or dry_run:
            return
        completed_sub_experiments.add(label)
        save_checkpoint(
            {
                "completed_groups": sorted(completed_groups),
                "group_b_completed_sub_experiments": sorted(completed_sub_experiments),
                "last_completed_step": f"Group B {label}",
            }
        )

    def should_skip(label):
        return resume and checkpoint_enabled and not dry_run and label in completed_sub_experiments

    minds_raw, datasets_by_rate = load_group_b_datasets()
    id2label = minds_raw.features["intent_class"].int2str
    sample_indices_5a = _select_sample_indices(len(minds_raw), 20)
    sample_indices_5b = sample_indices_5a[:10]
    baseline_index = sample_indices_5a[0]

    try:
        if should_skip("exp1_ga19a"):
            print("[GROUP B] Skipping Exp 1 GA19A dataset inspection (already completed)")
        else:
            print("[GROUP B] Exp 1 GA19A: Dataset inspection printout")
            display_dataset = minds_raw.remove_columns(
                [
                    column
                    for column in ["lang_id", "english_transcription"]
                    if column in minds_raw.column_names
                ]
            )
            example = minds_raw[0]
            audio_meta = example["audio"]
            intent_label = id2label(example["intent_class"])

            print(minds_raw)
            print(example)
            print(intent_label)
            print(display_dataset)
            print(
                "[GROUP B] Example 0 audio: "
                f"rate={audio_meta['sampling_rate']} Hz, "
                f"duration={len(audio_meta['array']) / audio_meta['sampling_rate']:.3f}s"
            )

            summary["dataset_inspections"] += 1
            mark_sub_experiment_complete("exp1_ga19a")

        if should_skip("exp1_ga19b"):
            print("[GROUP B] Skipping Exp 1 GA19B baseline (already completed)")
        else:
            print("[GROUP B] Exp 1 GA19B: Default ASR baseline")
            baseline_pipeline = None
            baseline_model_name = "default_pipeline"
            try:
                baseline_pipeline, baseline_model_name, resolved_device = load_group_b_asr_pipeline()
                print(
                    f"[GROUP B] Loaded baseline ASR pipeline: {baseline_model_name} "
                    f"on device {resolved_device}"
                )
                baseline_example = datasets_by_rate[16000][baseline_index]
                baseline_audio = baseline_example["audio"]
                baseline_result = run_ga19b(
                    baseline_pipeline,
                    audio_array=baseline_audio["array"],
                    sampling_rate=baseline_audio["sampling_rate"],
                    ground_truth=baseline_example["english_transcription"],
                    model_name=baseline_model_name,
                    sample_index=baseline_index,
                    csv_path=None,
                    sub_experiment="baseline",
                    extra_meta={"experiment": "exp1"},
                )
                append_csv_row(
                    exp1_csv,
                    [
                        "exp1",
                        "GA19B",
                        "automatic_speech_recognition",
                        GROUP_B_SAMPLE_SELECTION_SEED,
                        "",
                        "",
                        "",
                        baseline_model_name,
                        f"{GROUP_B_DATASET_ID}:{GROUP_B_DATASET_CONFIG}",
                        baseline_example["english_transcription"],
                        baseline_result["predicted"],
                        baseline_result["time"],
                        baseline_result["timestamp"],
                    ],
                )
                print(f"[GROUP B] Baseline transcription: {baseline_result['predicted']}")
                record_transcription("exp1", baseline_model_name)
            finally:
                if baseline_pipeline is not None:
                    cleanup_pipeline(baseline_pipeline, f"ASR baseline {baseline_model_name}")
                    baseline_pipeline = None
            mark_sub_experiment_complete("exp1_ga19b")

        comparison_models = GROUP_B_MODEL_COMPARISON
        comparison_sample_indices = sample_indices_5a[:1] if dry_run else sample_indices_5a
        for model_name in comparison_models:
            label = f"5a_{model_name.replace('/', '_').replace('-', '_')}"
            if should_skip(label):
                print(f"[GROUP B] Skipping Exp 5A for {model_name} (already completed)")
                continue

            print(f"[GROUP B] Exp 5A: Multi-model comparison with {model_name}")
            asr_pipeline = None
            resolved_model_name = model_name
            try:
                asr_pipeline, resolved_model_name, resolved_device = load_group_b_asr_pipeline(model_name)
                print(
                    f"[GROUP B] Loaded ASR pipeline: {resolved_model_name} "
                    f"on device {resolved_device}"
                )
                for sample_index in comparison_sample_indices:
                    sample = datasets_by_rate[16000][sample_index]
                    audio_meta = sample["audio"]
                    result = run_ga19b(
                        asr_pipeline,
                        audio_array=audio_meta["array"],
                        sampling_rate=audio_meta["sampling_rate"],
                        ground_truth=sample["english_transcription"],
                        model_name=resolved_model_name,
                        sample_index=sample_index,
                        csv_path=exp5_csv,
                        sub_experiment="5a",
                        extra_meta={"experiment": "exp5"},
                    )
                    print(
                        f"[GROUP B] 5A sample={sample_index} model={resolved_model_name} "
                        f"WER={result['wer']} CER={result['cer']}"
                    )
                    record_transcription("exp5", resolved_model_name)
            finally:
                if asr_pipeline is not None:
                    cleanup_pipeline(asr_pipeline, f"ASR {resolved_model_name}")
                    asr_pipeline = None
            mark_sub_experiment_complete(label)

        mismatch_models = [GROUP_B_MISMATCH_MODELS[0]] if dry_run else GROUP_B_MISMATCH_MODELS
        mismatch_rates = [8000] if dry_run else GROUP_B_MISMATCH_RATES
        mismatch_sample_indices = sample_indices_5b[:1] if dry_run else sample_indices_5b
        for model_tag, requested_model_name in mismatch_models:
            asr_pipeline = None
            resolved_model_name = requested_model_name or "default_pipeline"
            try:
                asr_pipeline, resolved_model_name, resolved_device = load_group_b_asr_pipeline(
                    requested_model_name
                )
                print(
                    f"[GROUP B] Loaded Exp 5B ASR pipeline: {resolved_model_name} "
                    f"on device {resolved_device}"
                )
                for target_rate in mismatch_rates:
                    label = f"5b_{model_tag}_{target_rate}"
                    if should_skip(label):
                        print(
                            f"[GROUP B] Skipping Exp 5B for {model_tag} at {target_rate} Hz "
                            "(already completed)"
                        )
                        continue

                    print(
                        f"[GROUP B] Exp 5B: Sampling-rate condition {target_rate} Hz "
                        f"with {resolved_model_name}"
                    )
                    dataset_for_rate = datasets_by_rate[target_rate]
                    for sample_index in mismatch_sample_indices:
                        sample = dataset_for_rate[sample_index]
                        audio_meta = sample["audio"]
                        result = run_ga19b(
                            asr_pipeline,
                            audio_array=audio_meta["array"],
                            sampling_rate=audio_meta["sampling_rate"],
                            ground_truth=sample["english_transcription"],
                            model_name=resolved_model_name,
                            sample_index=sample_index,
                            csv_path=exp5_csv,
                            sub_experiment="5b",
                            extra_meta={"experiment": "exp5"},
                        )
                        print(
                            f"[GROUP B] 5B sample={sample_index} model={resolved_model_name} "
                            f"rate={audio_meta['sampling_rate']}Hz WER={result['wer']} CER={result['cer']}"
                        )
                        record_transcription("exp5", resolved_model_name)
                    mark_sub_experiment_complete(label)
            finally:
                if asr_pipeline is not None:
                    cleanup_pipeline(asr_pipeline, f"ASR {resolved_model_name}")
                    asr_pipeline = None

        if checkpoint_enabled and not dry_run:
            completed_groups.add("B")
            save_checkpoint(
                {
                    "completed_groups": sorted(completed_groups),
                    "group_b_completed_sub_experiments": sorted(completed_sub_experiments),
                    "last_completed_step": "Group B complete",
                }
            )

        print(
            "[GROUP B] Complete. "
            f"Dataset inspections={summary['dataset_inspections']}, "
            f"transcriptions={summary['total_transcriptions']} "
            f"({summary['per_experiment']})."
        )
        return summary
    finally:
        summary["models_used"] = sorted(summary["models_used"])


def run_group_c(
    dry_run=False,
    experiments_dir=None,
    checkpoint_enabled=True,
    resume=False,
):
    """
    Group C: Bark TTS — Audio Generation Experiments.
    Exp 1 (GA19C baseline), Exp 6 (full), Exp 7 (GA19C seeds),
    Exp 6D (round-trip TTS -> ASR).
    """
    layout = _prepare_group_c_layout(experiments_dir or EXPERIMENTS_DIR)
    experiments_root = layout["root"]

    print(f"\n{'=' * 60}")
    print(f"[GROUP C] Starting {'dry-run' if dry_run else 'full'} Bark TTS workflow")
    print(f"[GROUP C] Output root: {experiments_root}")

    exp1_csv = init_csv(layout["exp1_dir"] / "exp1_baselines.csv", GROUP_A_EXP1_HEADER)
    exp6_csv = init_csv(layout["exp6_dir"] / "exp6_ga19c_tts.csv", GROUP_C_EXP6_HEADER)
    exp7_csv = init_csv(layout["exp7_dir"] / "exp7_seed_sensitivity.csv", GROUP_A_EXP7_HEADER)

    summary = {
        "mode": "dry-run" if dry_run else "full",
        "output_root": str(experiments_root),
        "tts_model": GROUP_C_TTS_MODEL,
        "best_asr_model": "",
        "round_trip_asr_model": "",
        "total_audio_files": 0,
        "round_trip_transcriptions": 0,
        "per_experiment": {"exp1": 0, "exp6": 0, "exp7": 0},
    }

    completed_sub_experiments = set()
    completed_groups = set()
    if checkpoint_enabled and resume and not dry_run:
        checkpoint = load_checkpoint()
        if checkpoint:
            completed_sub_experiments = set(checkpoint.get("group_c_completed_sub_experiments", []))
            completed_groups = set(checkpoint.get("completed_groups", []))
            print(
                "[GROUP C] Resuming from checkpoint. "
                f"Completed labels: {sorted(completed_sub_experiments)}"
            )

    def record_audio(experiment_key):
        summary["total_audio_files"] += 1
        summary["per_experiment"][experiment_key] += 1

    def mark_sub_experiment_complete(label):
        if not checkpoint_enabled or dry_run:
            return
        completed_sub_experiments.add(label)
        save_checkpoint(
            {
                "completed_groups": sorted(completed_groups),
                "group_c_completed_sub_experiments": sorted(completed_sub_experiments),
                "last_completed_step": f"Group C {label}",
            }
        )

    def should_skip(label):
        return resume and checkpoint_enabled and not dry_run and label in completed_sub_experiments

    best_asr_model = _select_best_group_b_asr_model()
    summary["best_asr_model"] = best_asr_model

    baseline_seed = 42
    exp6a_seeds = [baseline_seed] if dry_run else GROUP_C_EXP6A_SEEDS
    exp6b_texts = [GROUP_C_EXP6B_TEXTS[0]] if dry_run else GROUP_C_EXP6B_TEXTS
    exp6c_texts = [GROUP_C_EXP6C_TEXTS[1]] if dry_run else GROUP_C_EXP6C_TEXTS
    exp6d_texts = [GROUP_C_EXP6D_TEXTS[1]] if dry_run else GROUP_C_EXP6D_TEXTS
    exp6_variant_seeds = [baseline_seed] if dry_run else SEEDS_2
    exp7_seeds = [baseline_seed] if dry_run else SEEDS_8

    tts_pipeline = None
    resolved_tts_model = GROUP_C_TTS_MODEL
    try:
        tts_pipeline, resolved_tts_model, resolved_device = load_group_c_tts_pipeline()
        summary["tts_model"] = resolved_tts_model
        print(
            f"[GROUP C] Loaded Bark pipeline: {resolved_tts_model} "
            f"on device {resolved_device}"
        )

        if should_skip("exp1_ga19c"):
            print("[GROUP C] Skipping Exp 1 GA19C baseline (already completed)")
        else:
            print("[GROUP C] Exp 1 GA19C: Bark baseline")
            baseline_result = run_ga19c(
                tts_pipeline,
                input_text=GA19C_DEFAULT_TEXT,
                seed=baseline_seed,
                save_dir=layout["exp1_dir"],
                csv_path=None,
                sub_experiment="baseline",
                extra_meta={
                    "experiment": "exp1",
                    "model_name": resolved_tts_model,
                    "text_tag": "default",
                    "text_category": "default",
                },
            )
            append_csv_row(
                exp1_csv,
                [
                    "exp1",
                    "GA19C",
                    "text_to_speech",
                    baseline_seed,
                    "",
                    "",
                    "",
                    resolved_tts_model,
                    "",
                    GA19C_DEFAULT_TEXT,
                    _relative_output_path(baseline_result["path"], experiments_root),
                    baseline_result["time"],
                    baseline_result["timestamp"],
                ],
            )
            print(
                f"[GROUP C] Baseline audio saved: "
                f"{_relative_output_path(baseline_result['path'], experiments_root)}"
            )
            record_audio("exp1")
            mark_sub_experiment_complete("exp1_ga19c")

        if should_skip("6a"):
            print("[GROUP C] Skipping Exp 6A repeated-generation sweep (already completed)")
        else:
            print(f"[GROUP C] Exp 6A: repeated generation across {len(exp6a_seeds)} seed(s)")
            for seed in exp6a_seeds:
                result = run_ga19c(
                    tts_pipeline,
                    input_text=GA19C_DEFAULT_TEXT,
                    seed=seed,
                    save_dir=layout["exp6_dir"],
                    csv_path=exp6_csv,
                    sub_experiment="6a",
                    extra_meta={
                        "experiment": "exp6",
                        "model_name": resolved_tts_model,
                        "text_tag": "repeat",
                        "text_category": "repeat",
                    },
                )
                print(
                    f"[GROUP C] 6A seed={seed} duration={result['duration']:.3f}s "
                    f"time={result['time']:.3f}s"
                )
                record_audio("exp6")
            mark_sub_experiment_complete("6a")

        for config in exp6b_texts:
            label = f"6b_{config['text_tag']}"
            if should_skip(label):
                print(f"[GROUP C] Skipping Exp 6B {config['text_tag']} (already completed)")
                continue

            print(f"[GROUP C] Exp 6B: text length = {config['text_tag']}")
            for seed in exp6_variant_seeds:
                result = run_ga19c(
                    tts_pipeline,
                    input_text=config["input_text"],
                    seed=seed,
                    save_dir=layout["exp6_dir"],
                    csv_path=exp6_csv,
                    sub_experiment="6b",
                    extra_meta={
                        "experiment": "exp6",
                        "model_name": resolved_tts_model,
                        "text_tag": config["text_tag"],
                        "text_category": config["text_category"],
                    },
                )
                print(
                    f"[GROUP C] 6B text={config['text_tag']} seed={seed} "
                    f"duration={result['duration']:.3f}s"
                )
                record_audio("exp6")
            mark_sub_experiment_complete(label)

        for config in exp6c_texts:
            label = f"6c_{config['text_tag']}"
            if should_skip(label):
                print(f"[GROUP C] Skipping Exp 6C {config['text_tag']} (already completed)")
                continue

            print(f"[GROUP C] Exp 6C: text domain = {config['text_tag']}")
            for seed in exp6_variant_seeds:
                result = run_ga19c(
                    tts_pipeline,
                    input_text=config["input_text"],
                    seed=seed,
                    save_dir=layout["exp6_dir"],
                    csv_path=exp6_csv,
                    sub_experiment="6c",
                    extra_meta={
                        "experiment": "exp6",
                        "model_name": resolved_tts_model,
                        "text_tag": config["text_tag"],
                        "text_category": config["text_category"],
                    },
                )
                print(
                    f"[GROUP C] 6C text={config['text_tag']} seed={seed} "
                    f"duration={result['duration']:.3f}s"
                )
                record_audio("exp6")
            mark_sub_experiment_complete(label)

        pending_round_trip = [
            config for config in exp6d_texts if not should_skip(f"6d_{config['text_tag']}")
        ]
        if pending_round_trip:
            asr_pipeline = None
            resolved_round_trip_model = best_asr_model
            try:
                asr_pipeline, resolved_round_trip_model, resolved_asr_device = load_group_b_asr_pipeline(
                    best_asr_model
                )
                summary["round_trip_asr_model"] = resolved_round_trip_model
                print(
                    f"[GROUP C] Loaded round-trip ASR pipeline: {resolved_round_trip_model} "
                    f"on device {resolved_asr_device}"
                )
                for config in pending_round_trip:
                    label = f"6d_{config['text_tag']}"
                    print(f"[GROUP C] Exp 6D: round-trip text = {config['text_tag']}")
                    tts_result = run_ga19c(
                        tts_pipeline,
                        input_text=config["input_text"],
                        seed=baseline_seed,
                        save_dir=layout["exp6_dir"],
                        csv_path=None,
                        sub_experiment="6d",
                        extra_meta={
                            "experiment": "exp6",
                            "model_name": resolved_tts_model,
                            "text_tag": config["text_tag"],
                            "text_category": config["text_category"],
                        },
                    )
                    asr_audio = _resample_audio_for_asr(
                        tts_result["audio"],
                        tts_result["sampling_rate"],
                        target_rate=16000,
                    )
                    asr_result = run_ga19b(
                        asr_pipeline,
                        audio_array=asr_audio,
                        sampling_rate=16000,
                        ground_truth=config["input_text"],
                        model_name=resolved_round_trip_model,
                        sample_index=-1,
                        csv_path=None,
                        sub_experiment="6d",
                        extra_meta={"experiment": "exp6"},
                    )
                    append_csv_row(
                        exp6_csv,
                        [
                            "exp6",
                            "6d",
                            "GA19C",
                            baseline_seed,
                            resolved_tts_model,
                            config["input_text"],
                            len(config["input_text"].split()),
                            config["text_category"],
                            str(tts_result["path"].name),
                            round(tts_result["duration"], 3),
                            tts_result["sampling_rate"],
                            round(tts_result["stats"]["rms"], 6),
                            tts_result["time"],
                            asr_result["predicted"],
                            asr_result["wer"],
                            tts_result["timestamp"],
                        ],
                    )
                    print(
                        f"[GROUP C] 6D text={config['text_tag']} model={resolved_round_trip_model} "
                        f"WER={asr_result['wer']} CER={asr_result['cer']}"
                    )
                    record_audio("exp6")
                    summary["round_trip_transcriptions"] += 1
                    mark_sub_experiment_complete(label)
            finally:
                if asr_pipeline is not None:
                    cleanup_pipeline(asr_pipeline, f"ASR round-trip {resolved_round_trip_model}")
                    asr_pipeline = None
        elif not summary["round_trip_asr_model"]:
            summary["round_trip_asr_model"] = best_asr_model

        if should_skip("exp7_ga19c"):
            print("[GROUP C] Skipping Exp 7 GA19C seed sensitivity (already completed)")
        else:
            print(f"[GROUP C] Exp 7 GA19C: seed sensitivity across {len(exp7_seeds)} seed(s)")
            for seed in exp7_seeds:
                result = run_ga19c(
                    tts_pipeline,
                    input_text=GA19C_DEFAULT_TEXT,
                    seed=seed,
                    save_dir=layout["exp7_dir"],
                    csv_path=None,
                    sub_experiment="seed",
                    extra_meta={
                        "experiment": "exp7",
                        "model_name": resolved_tts_model,
                        "text_tag": "default",
                        "text_category": "seed_sensitivity",
                    },
                )
                append_csv_row(
                    exp7_csv,
                    [
                        "exp7",
                        "GA19C",
                        "text_to_speech",
                        seed,
                        GA19C_DEFAULT_TEXT,
                        json.dumps(
                            {
                                "model": resolved_tts_model,
                                "text_category": "seed_sensitivity",
                                "sub_experiment": "seed",
                            },
                            sort_keys=True,
                        ),
                        _relative_output_path(result["path"], experiments_root),
                        "audio",
                        result["time"],
                        result["timestamp"],
                    ],
                )
                print(
                    f"[GROUP C] Exp 7 seed={seed} duration={result['duration']:.3f}s "
                    f"time={result['time']:.3f}s"
                )
                record_audio("exp7")
            mark_sub_experiment_complete("exp7_ga19c")

        if checkpoint_enabled and not dry_run:
            completed_groups.add("C")
            save_checkpoint(
                {
                    "completed_groups": sorted(completed_groups),
                    "group_c_completed_sub_experiments": sorted(completed_sub_experiments),
                    "last_completed_step": "Group C complete",
                }
            )

        print(
            "[GROUP C] Complete. "
            f"Audio files={summary['total_audio_files']}, "
            f"round-trip transcriptions={summary['round_trip_transcriptions']} "
            f"({summary['per_experiment']})."
        )
        return summary
    finally:
        if tts_pipeline is not None:
            cleanup_pipeline(tts_pipeline, f"Bark TTS {resolved_tts_model}")
            tts_pipeline = None


if __name__ == "__main__":
    main()