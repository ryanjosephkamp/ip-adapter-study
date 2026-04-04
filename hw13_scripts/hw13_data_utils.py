"""
HW13 Data Utilities — CSV logging, image saving, audio saving, timing,
console capture, WER/CER computation, and checkpointing.
Used by kamp_hw13.py and hw13_experiment_runner.py.
"""

import csv
import json
import re
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


# --- Project paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "hw13_experiments"
PRINTOUTS_DIR = PROJECT_ROOT / "hw13_printouts"
CACHE_DIR = EXPERIMENTS_DIR / "cached_inputs"
CHECKPOINT_PATH = EXPERIMENTS_DIR / "checkpoint.json"


# --- Timing context manager ---
@contextmanager
def timer(label=""):
    """Context manager that measures wall-clock time in seconds."""
    start = time.perf_counter()
    result = {"elapsed_sec": 0.0}
    try:
        yield result
    finally:
        result["elapsed_sec"] = round(time.perf_counter() - start, 3)
        if label:
            print(f"[TIMER] {label}: {result['elapsed_sec']:.3f}s")


# --- CSV utilities ---
def init_csv(filepath, header_row):
    """Create a CSV file with a header row if it does not exist."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if not filepath.exists():
        with filepath.open("w", newline="", encoding="utf-8") as file_handle:
            writer = csv.writer(file_handle)
            writer.writerow(header_row)
        print(f"[CSV] Initialized: {filepath}")
    return filepath


def append_csv_row(filepath, row_data):
    """Append a single data row to a CSV file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("a", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(row_data)


def count_csv_rows(filepath):
    """Count data rows in a CSV file, excluding the header."""
    filepath = Path(filepath)
    if not filepath.exists():
        return 0
    with filepath.open("r", encoding="utf-8") as file_handle:
        return max(sum(1 for _ in file_handle) - 1, 0)


# --- Image saving ---
def save_image(image, filepath, label=""):
    """Save a PIL Image to disk."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    image.save(filepath)
    size_kb = filepath.stat().st_size / 1024
    if label:
        print(f"[IMG] {label}: {filepath.name} ({size_kb:.1f} KB)")
    return filepath


def load_cached_image(name):
    """Load a cached input image by name from the cache directory."""
    path = CACHE_DIR / f"{name}.png"
    if not path.exists():
        raise FileNotFoundError(f"Cached image not found: {path}")
    return Image.open(path).convert("RGB")


# --- Audio saving ---
def save_audio(audio_array, sampling_rate, filepath, label=""):
    """Save an audio waveform to a WAV file using scipy."""
    import scipy.io.wavfile

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    audio_np = np.asarray(audio_array)
    audio_np = np.squeeze(audio_np)
    if audio_np.dtype not in (np.int16, np.int32, np.uint8, np.float32, np.float64):
        audio_np = audio_np.astype(np.float32)
    elif audio_np.dtype == np.float64:
        audio_np = audio_np.astype(np.float32)

    scipy.io.wavfile.write(str(filepath), rate=int(sampling_rate), data=audio_np)
    size_kb = filepath.stat().st_size / 1024
    if label:
        print(f"[AUDIO] {label}: {filepath.name} ({size_kb:.1f} KB)")
    return filepath


def get_audio_stats(audio_array):
    """Compute basic waveform statistics for an audio array."""
    audio_np = np.asarray(audio_array).astype(np.float64).flatten()
    if audio_np.size == 0:
        return {
            "duration_samples": 0,
            "rms": 0.0,
            "max_amplitude": 0.0,
            "mean_amplitude": 0.0,
        }

    return {
        "duration_samples": int(audio_np.size),
        "rms": float(np.sqrt(np.mean(audio_np ** 2))),
        "max_amplitude": float(np.max(np.abs(audio_np))),
        "mean_amplitude": float(np.mean(np.abs(audio_np))),
    }


# --- WER/CER computation ---
def compute_wer_cer(reference, hypothesis):
    """
    Compute Word Error Rate and Character Error Rate using jiwer.
    Strings are normalized to lowercase alphanumeric text before comparison.
    """
    try:
        from jiwer import cer as jiwer_cer
        from jiwer import wer as jiwer_wer
    except ImportError:
        print("[WARN] jiwer not installed. Returning -1 for WER/CER.")
        return -1.0, -1.0

    def normalize(text):
        normalized = str(text).lower().strip()
        normalized = re.sub(r"[^\w\s]", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    ref_norm = normalize(reference)
    hyp_norm = normalize(hypothesis)

    if not ref_norm:
        if not hyp_norm:
            return 0.0, 0.0
        return 1.0, 1.0

    wer_value = jiwer_wer(ref_norm, hyp_norm)
    cer_value = jiwer_cer(ref_norm, hyp_norm)
    return round(wer_value, 4), round(cer_value, 4)


# --- Console capture ---
class TeeLogger:
    """Duplicate stdout and stderr to both the console and a log file."""

    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path.open("a", encoding="utf-8")
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def start(self):
        sys.stdout = self
        sys.stderr = self
        self._write_header()
        return self

    def _write_header(self):
        header = f"\n{'=' * 60}\n[SESSION START] {datetime.now().isoformat()}\n{'=' * 60}\n"
        self.log_file.write(header)
        self.log_file.flush()

    def write(self, data):
        self.stdout.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.stdout.flush()
        self.log_file.flush()

    def isatty(self):
        stdout_isatty = getattr(self.stdout, "isatty", None)
        if callable(stdout_isatty):
            return bool(stdout_isatty())
        return False

    def fileno(self):
        stdout_fileno = getattr(self.stdout, "fileno", None)
        if callable(stdout_fileno):
            return stdout_fileno()
        raise OSError("Underlying stdout does not expose fileno()")

    @property
    def encoding(self):
        return getattr(self.stdout, "encoding", "utf-8")

    def __getattr__(self, name):
        return getattr(self.stdout, name)

    def stop(self):
        footer = f"\n{'=' * 60}\n[SESSION END] {datetime.now().isoformat()}\n{'=' * 60}\n"
        self.log_file.write(footer)
        self.log_file.flush()
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.log_file.close()


# --- Checkpointing ---
def save_checkpoint(data):
    """Save checkpoint metadata to JSON."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    payload["timestamp"] = datetime.now().isoformat()
    with CHECKPOINT_PATH.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)
    print(f"[CHECKPOINT] Saved: {payload.get('last_completed_step', 'unknown')}")


def load_checkpoint():
    """Load checkpoint metadata, or return None if no checkpoint exists."""
    if not CHECKPOINT_PATH.exists():
        return None
    with CHECKPOINT_PATH.open("r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    print(f"[CHECKPOINT] Loaded: last completed = {data.get('last_completed_step', 'unknown')}")
    return data


# --- Timestamp utility ---
def now_iso():
    """Return the current local timestamp in project log format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# --- VRAM cleanup ---
def cleanup_pipeline(pipeline, pipeline_name=""):
    """Delete a pipeline object and clear CUDA cache if torch is available."""
    import gc

    del pipeline
    gc.collect()

    try:
        import torch
    except ImportError:
        if pipeline_name:
            print(f"[VRAM] Unloaded without torch cleanup: {pipeline_name}")
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if pipeline_name:
        print(f"[VRAM] Unloaded: {pipeline_name}")