"""Step 7 cross-experiment analysis and summary-statistics generation."""

from __future__ import annotations

import json
from pathlib import Path

from hw13_data_utils import now_iso
from hw13_figure_gen import (
    EXP1_DIR,
    EXP2_DIR,
    EXP3_DIR,
    EXP4_DIR,
    EXP5_DIR,
    EXP6_DIR,
    EXP7_DIR,
    EXP_DIR,
    PROJECT_ROOT,
    collect_timing_samples,
    collect_timing_summary,
    copy_report_figures,
    figure_28_cross_modal_comparison_table,
    figure_29_cross_timing_comparison,
    get_cross_modal_comparison_records,
    read_csv_rows,
    to_float,
)


CSV_PATHS = {
    "exp1_baselines": EXP1_DIR / "exp1_baselines.csv",
    "exp2_ga18b_scale": EXP2_DIR / "exp2_ga18b_scale.csv",
    "exp3_ga18c_blocks": EXP3_DIR / "exp3_ga18c_blocks.csv",
    "exp4_text_image_interaction": EXP4_DIR / "exp4_text_image_interaction.csv",
    "exp5_ga19b_asr": EXP5_DIR / "exp5_ga19b_asr.csv",
    "exp6_ga19c_tts": EXP6_DIR / "exp6_ga19c_tts.csv",
    "exp7_seed_sensitivity": EXP7_DIR / "exp7_seed_sensitivity.csv",
}


EXPERIMENT_DIRS = {
    "exp1_baselines": EXP1_DIR,
    "exp2_ga18b_scale": EXP2_DIR,
    "exp3_ga18c_blocks": EXP3_DIR,
    "exp4_text_image_interaction": EXP4_DIR,
    "exp5_ga19b_asr": EXP5_DIR,
    "exp6_ga19c_tts": EXP6_DIR,
    "exp7_seed_sensitivity": EXP7_DIR,
}


def round_or_none(value, digits=3):
    if value is None:
        return None
    return round(float(value), digits)


def resolve_output_path(experiment_name, output_value):
    output_text = str(output_value or "").strip()
    if not output_text:
        return None

    output_path = Path(output_text)
    if output_path.is_absolute():
        return output_path

    exp_relative_path = EXPERIMENT_DIRS[experiment_name] / output_path
    if exp_relative_path.exists():
        return exp_relative_path

    experiments_relative_path = EXP_DIR / output_path
    if experiments_relative_path.exists():
        return experiments_relative_path

    return exp_relative_path


def load_experiment_rows():
    rows_by_experiment = {}
    for experiment_name, csv_path in CSV_PATHS.items():
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing required CSV: {csv_path}")
        rows_by_experiment[experiment_name] = read_csv_rows(csv_path)
    return rows_by_experiment


def summarize_times(values):
    valid_values = [float(value) for value in values if value is not None]
    if not valid_values:
        return {
            "sample_count": 0,
            "mean_sec": None,
            "min_sec": None,
            "max_sec": None,
        }

    return {
        "sample_count": len(valid_values),
        "mean_sec": round_or_none(sum(valid_values) / len(valid_values)),
        "min_sec": round_or_none(min(valid_values)),
        "max_sec": round_or_none(max(valid_values)),
    }


def collect_output_counts(rows_by_experiment):
    image_paths = set()
    audio_paths = set()

    for row in rows_by_experiment["exp1_baselines"]:
        script = row.get("script")
        if script in {"GA18B", "GA18C"}:
            output_path = resolve_output_path("exp1_baselines", row.get("output_path"))
            if output_path and output_path.exists():
                image_paths.add(output_path.resolve())
        elif script == "GA19C":
            output_path = resolve_output_path("exp1_baselines", row.get("output_path"))
            if output_path and output_path.exists():
                audio_paths.add(output_path.resolve())

    for experiment_name in ("exp2_ga18b_scale", "exp3_ga18c_blocks", "exp4_text_image_interaction"):
        for row in rows_by_experiment[experiment_name]:
            output_path = resolve_output_path(experiment_name, row.get("image_path"))
            if output_path and output_path.exists():
                image_paths.add(output_path.resolve())

    for row in rows_by_experiment["exp6_ga19c_tts"]:
        output_path = resolve_output_path("exp6_ga19c_tts", row.get("output_path"))
        if output_path and output_path.exists():
            audio_paths.add(output_path.resolve())

    for row in rows_by_experiment["exp7_seed_sensitivity"]:
        output_path = resolve_output_path("exp7_seed_sensitivity", row.get("output_path"))
        if not output_path or not output_path.exists():
            continue
        if row.get("output_type") == "image":
            image_paths.add(output_path.resolve())
        elif row.get("output_type") == "audio":
            audio_paths.add(output_path.resolve())

    total_transcriptions = 0
    total_transcriptions += sum(
        1
        for row in rows_by_experiment["exp1_baselines"]
        if row.get("script") == "GA19B" and str(row.get("output_path", "")).strip()
    )
    total_transcriptions += sum(
        1
        for row in rows_by_experiment["exp5_ga19b_asr"]
        if str(row.get("predicted_transcription", "")).strip()
    )
    total_transcriptions += sum(
        1
        for row in rows_by_experiment["exp6_ga19c_tts"]
        if str(row.get("round_trip_transcription", "")).strip()
    )

    return {
        "total_images_generated": len(image_paths),
        "total_audio_files_generated": len(audio_paths),
        "total_transcriptions": total_transcriptions,
    }


def collect_total_time_seconds(rows_by_experiment):
    total_time = 0.0
    for row in rows_by_experiment["exp1_baselines"]:
        total_time += to_float(row.get("generation_time_sec"), default=0.0)
    for experiment_name in (
        "exp2_ga18b_scale",
        "exp3_ga18c_blocks",
        "exp4_text_image_interaction",
        "exp6_ga19c_tts",
        "exp7_seed_sensitivity",
    ):
        for row in rows_by_experiment[experiment_name]:
            total_time += to_float(row.get("generation_time_sec"), default=0.0)
    for row in rows_by_experiment["exp5_ga19b_asr"]:
        total_time += to_float(row.get("inference_time_sec"), default=0.0)
    return round_or_none(total_time)


def build_summary(rows_by_experiment):
    timing_samples = collect_timing_samples()
    timing_summary = collect_timing_summary()
    output_counts = collect_output_counts(rows_by_experiment)

    per_paradigm_timing_details = {
        paradigm: summarize_times(values)
        for paradigm, values in timing_samples.items()
    }

    experiment_completion = {
        experiment_name: {
            "completed": bool(rows),
            "csv_path": str(csv_path.relative_to(PROJECT_ROOT)),
            "csv_rows": len(rows),
        }
        for experiment_name, (csv_path, rows) in (
            (name, (path, rows_by_experiment[name]))
            for name, path in CSV_PATHS.items()
        )
    }

    return {
        **output_counts,
        "total_gpu_time_sec": collect_total_time_seconds(rows_by_experiment),
        "per_paradigm_timing": {
            paradigm: round_or_none(mean_time)
            for paradigm, mean_time in timing_summary.items()
        },
        "per_paradigm_timing_details": per_paradigm_timing_details,
        "cross_paradigm_comparison": get_cross_modal_comparison_records(),
        "experiment_completion": experiment_completion,
        "source_csvs": {
            experiment_name: str(csv_path.relative_to(PROJECT_ROOT))
            for experiment_name, csv_path in CSV_PATHS.items()
        },
        "generated_at": now_iso(),
    }


def write_summary_statistics(summary):
    summary_path = EXP_DIR / "summary_statistics.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary_path


def generate_cross_figures():
    generated_paths = []
    figure_28_cross_modal_comparison_table(generated_paths)
    figure_29_cross_timing_comparison(generated_paths)
    copy_report_figures(generated_paths)
    return generated_paths


def main():
    rows_by_experiment = load_experiment_rows()
    generated_paths = generate_cross_figures()
    summary = build_summary(rows_by_experiment)
    summary_path = write_summary_statistics(summary)

    print(f"[STEP 7] Cross figures regenerated: {len(generated_paths)}")
    print(f"[STEP 7] Summary statistics saved: {summary_path.relative_to(PROJECT_ROOT)}")
    print(
        "[STEP 7] Output totals: "
        f"{summary['total_images_generated']} images, "
        f"{summary['total_audio_files_generated']} audio files, "
        f"{summary['total_transcriptions']} transcriptions"
    )


if __name__ == "__main__":
    main()