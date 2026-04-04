"""
HW13 Figure Generation — programmatic generation of the planned figure set.

Reads experiment images, CSV files, and audio files from hw13_experiments/
and generates publication-ready figures for the report and LaTeX outputs.
"""

from __future__ import annotations

import csv
import shutil
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.io import wavfile


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = PROJECT_ROOT / "hw13_experiments"
REPORT_FIGS = PROJECT_ROOT / "hw13_reports" / "figures"
LATEX_FIGS = PROJECT_ROOT / "hw13_reports" / "latex" / "figures"

EXP1_DIR = EXP_DIR / "exp1_baselines"
EXP2_DIR = EXP_DIR / "exp2_ga18b_scale"
EXP3_DIR = EXP_DIR / "exp3_ga18c_blocks"
EXP4_DIR = EXP_DIR / "exp4_text_image_interaction"
EXP5_DIR = EXP_DIR / "exp5_ga19b_asr"
EXP6_DIR = EXP_DIR / "exp6_ga19c_tts"
EXP7_DIR = EXP_DIR / "exp7_seed_sensitivity"

STYLE = {
    "dpi": 300,
    "title_size": 14,
    "label_size": 11,
    "tick_size": 9,
    "grid_alpha": 0.28,
    "line_width": 1.0,
    "colors": [
        "#1f77b4",
        "#2ca02c",
        "#ff7f0e",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#17becf",
        "#7f7f7f",
    ],
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": STYLE["tick_size"],
        "axes.titlesize": STYLE["label_size"],
        "axes.labelsize": STYLE["label_size"],
        "figure.titlesize": STYLE["title_size"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def ensure_directories():
    REPORT_FIGS.mkdir(parents=True, exist_ok=True)
    LATEX_FIGS.mkdir(parents=True, exist_ok=True)


def read_csv_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value, default=np.nan):
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def load_image_array(path: Path):
    if not Path(path).exists():
        return None
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def read_audio_mono(path: Path):
    sampling_rate, audio = wavfile.read(path)
    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32)
    max_abs = np.max(np.abs(audio)) if audio.size else 0.0
    if max_abs > 0:
        audio = audio / max_abs
    return sampling_rate, audio


def save_figure(fig, save_path: Path, generated_paths: list[Path]):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    generated_paths.append(save_path)
    print(f"[FIGURE] Saved: {save_path.relative_to(PROJECT_ROOT)}")


def copy_report_figures(generated_paths: list[Path]):
    for fig_path in generated_paths:
        shutil.copy2(fig_path, REPORT_FIGS / fig_path.name)
        shutil.copy2(fig_path, LATEX_FIGS / fig_path.name)
    print(f"[FIGURE] Copied {len(generated_paths)} figure(s) to report directories.")


def add_missing_panel(ax, label="Missing"):
    ax.set_facecolor("#f2f2f2")
    ax.text(0.5, 0.5, label, ha="center", va="center", color="gray")
    ax.set_xticks([])
    ax.set_yticks([])


def create_image_grid(image_paths, row_labels, col_labels, title, save_path, generated_paths):
    nrows = len(row_labels)
    ncols = len(col_labels)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 2.7, nrows * 2.7),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=STYLE["title_size"], fontweight="bold", y=1.01)

    for row_idx in range(nrows):
        for col_idx in range(ncols):
            ax = axes[row_idx][col_idx]
            image = load_image_array(image_paths[row_idx][col_idx])
            if image is None:
                add_missing_panel(ax)
            else:
                ax.imshow(image)
                ax.set_xticks([])
                ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(col_labels[col_idx], fontsize=STYLE["tick_size"])
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx], fontsize=STYLE["tick_size"], rotation=0, labelpad=50, va="center")

    plt.tight_layout()
    save_figure(fig, save_path, generated_paths)


def create_gallery(image_paths, labels, title, save_path, generated_paths, ncols=4):
    n_items = len(image_paths)
    nrows = int(np.ceil(n_items / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 2.8, nrows * 2.9),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=STYLE["title_size"], fontweight="bold", y=1.01)

    flat_axes = axes.flatten()
    for idx, ax in enumerate(flat_axes):
        if idx >= n_items:
            ax.axis("off")
            continue
        image = load_image_array(image_paths[idx])
        if image is None:
            add_missing_panel(ax)
        else:
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(labels[idx], fontsize=STYLE["tick_size"])

    plt.tight_layout()
    save_figure(fig, save_path, generated_paths)


def create_bar_chart(categories, values, title, xlabel, ylabel, save_path, generated_paths, colors=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = colors or STYLE["colors"]
    bars = ax.bar(categories, values, color=palette[: len(categories)], edgecolor="black", alpha=0.85)
    max_value = max(values) if values else 0.0
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + max(max_value * 0.015, 0.01),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=STYLE["tick_size"],
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=STYLE["grid_alpha"])
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    save_figure(fig, save_path, generated_paths)


def create_grouped_bar_chart(categories, series_map, title, xlabel, ylabel, save_path, generated_paths):
    fig, ax = plt.subplots(figsize=(10, 6))
    x_positions = np.arange(len(categories))
    n_series = len(series_map)
    width = 0.36 if n_series == 2 else 0.8 / max(n_series, 1)

    for idx, (series_name, values) in enumerate(series_map.items()):
        offset = (idx - (n_series - 1) / 2.0) * width
        bars = ax.bar(
            x_positions + offset,
            values,
            width=width,
            label=series_name,
            color=STYLE["colors"][idx % len(STYLE["colors"])],
            edgecolor="black",
            alpha=0.82,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + max(max(values) * 0.015, 0.01),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=STYLE["tick_size"],
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(categories)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=STYLE["grid_alpha"])
    plt.tight_layout()
    save_figure(fig, save_path, generated_paths)


def create_box_plot(data_map, title, xlabel, ylabel, save_path, generated_paths):
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = list(data_map.keys())
    data = [data_map[label] for label in labels]
    box_plot = ax.boxplot(data, patch_artist=True, labels=labels)
    for idx, patch in enumerate(box_plot["boxes"]):
        patch.set_facecolor(STYLE["colors"][idx % len(STYLE["colors"])])
        patch.set_alpha(0.75)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=STYLE["grid_alpha"])
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    save_figure(fig, save_path, generated_paths)


def create_line_series_plot(series_map, title, xlabel, ylabel, save_path, generated_paths):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for idx, (series_name, x_values, y_values) in enumerate(series_map):
        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=STYLE["line_width"] + 0.4,
            label=series_name,
            color=STYLE["colors"][idx % len(STYLE["colors"])],
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=STYLE["grid_alpha"])
    ax.legend()
    plt.tight_layout()
    save_figure(fig, save_path, generated_paths)


def create_scatter_plot(points_by_group, title, xlabel, ylabel, save_path, generated_paths):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for idx, (group_name, points) in enumerate(points_by_group.items()):
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        ax.scatter(
            x_values,
            y_values,
            s=60,
            alpha=0.85,
            color=STYLE["colors"][idx % len(STYLE["colors"])],
            edgecolors="black",
            linewidths=0.4,
            label=group_name,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=STYLE["grid_alpha"])
    ax.legend()
    plt.tight_layout()
    save_figure(fig, save_path, generated_paths)


def create_waveform_comparison(audio_paths, labels, title, save_path, generated_paths):
    fig, axes = plt.subplots(len(audio_paths), 1, figsize=(12, max(7, 2.1 * len(audio_paths))), squeeze=False)
    fig.suptitle(title, fontsize=STYLE["title_size"], fontweight="bold", y=1.01)

    for idx, (audio_path, label) in enumerate(zip(audio_paths, labels)):
        ax = axes[idx][0]
        if not Path(audio_path).exists():
            add_missing_panel(ax)
            ax.set_ylabel(label, rotation=0, labelpad=55, va="center")
            continue
        sampling_rate, audio = read_audio_mono(audio_path)
        time_axis = np.arange(audio.size) / float(sampling_rate)
        ax.plot(time_axis, audio, color=STYLE["colors"][idx % len(STYLE["colors"])], linewidth=0.7)
        ax.set_ylabel(label, rotation=0, labelpad=55, va="center")
        ax.grid(True, alpha=STYLE["grid_alpha"])
        if idx == len(audio_paths) - 1:
            ax.set_xlabel("Time (seconds)")
        else:
            ax.set_xticklabels([])

    plt.tight_layout()
    save_figure(fig, save_path, generated_paths)


def create_tts_seed_summary(audio_paths, seed_labels, generation_times, save_path, generated_paths):
    fig = plt.figure(figsize=(12, 9))
    fig.suptitle("Experiment 7: Bark TTS Seed Sensitivity", fontsize=STYLE["title_size"], fontweight="bold", y=1.01)
    grid = fig.add_gridspec(2, 1, height_ratios=[2.3, 1.0])

    ax_wave = fig.add_subplot(grid[0, 0])
    ax_bar = fig.add_subplot(grid[1, 0])

    duration_values = []
    for idx, (audio_path, label) in enumerate(zip(audio_paths, seed_labels)):
        sampling_rate, audio = read_audio_mono(audio_path)
        duration_values.append(audio.size / float(sampling_rate))
        time_axis = np.arange(audio.size) / float(sampling_rate)
        offset = idx * 2.2
        ax_wave.plot(time_axis, audio + offset, color=STYLE["colors"][idx % len(STYLE["colors"])], linewidth=0.7)
        ax_wave.text(0.02, offset + 0.7, label, fontsize=STYLE["tick_size"], va="center")

    ax_wave.set_xlabel("Time (seconds)")
    ax_wave.set_ylabel("Seed-offset waveform")
    ax_wave.set_title("Normalized Bark waveforms by seed")
    ax_wave.grid(True, alpha=STYLE["grid_alpha"])

    x_positions = np.arange(len(seed_labels))
    width = 0.38
    ax_bar.bar(x_positions - width / 2.0, duration_values, width=width, label="Audio duration (s)", color=STYLE["colors"][0], edgecolor="black", alpha=0.82)
    ax_bar.bar(x_positions + width / 2.0, generation_times, width=width, label="Generation time (s)", color=STYLE["colors"][2], edgecolor="black", alpha=0.82)
    ax_bar.set_xticks(x_positions)
    ax_bar.set_xticklabels(seed_labels, rotation=15, ha="right")
    ax_bar.set_ylabel("Seconds")
    ax_bar.set_title("Duration and generation time by seed")
    ax_bar.grid(axis="y", alpha=STYLE["grid_alpha"])
    ax_bar.legend()

    plt.tight_layout()
    save_figure(fig, save_path, generated_paths)


def create_table_figure(
    col_labels,
    rows,
    title,
    save_path,
    generated_paths,
    figsize=(13, 4.8),
    font_size=9,
    y_scale=1.65,
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_title(title, fontweight="bold", pad=18)
    table = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, y_scale)
    if hasattr(table, "auto_set_column_width"):
        table.auto_set_column_width(col=list(range(len(col_labels))))

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#dbe9f6")
        else:
            cell.set_facecolor("#f7f7f7" if row_idx % 2 == 0 else "white")

    plt.tight_layout()
    save_figure(fig, save_path, generated_paths)


def compute_mean(values):
    filtered = [value for value in values if not np.isnan(value)]
    if not filtered:
        return float("nan")
    return float(sum(filtered) / len(filtered))


def categorize_computational_cost(mean_time_seconds):
    if np.isnan(mean_time_seconds):
        return "unknown"
    if mean_time_seconds < 0.5:
        return "low"
    if mean_time_seconds < 5.0:
        return "moderate"
    if mean_time_seconds < 15.0:
        return "high"
    return "very high"


def figure_01_baseline_gallery(generated_paths):
    exp1_rows = read_csv_rows(EXP1_DIR / "exp1_baselines.csv")
    ga19b_row = next((row for row in exp1_rows if row.get("script") == "GA19B"), None)

    fig = plt.figure(figsize=(12, 10))
    fig.suptitle("Experiment 1: Baseline Gallery", fontsize=STYLE["title_size"], fontweight="bold", y=1.01)
    grid = fig.add_gridspec(2, 2)

    ax_ga18b = fig.add_subplot(grid[0, 0])
    ax_ga18c = fig.add_subplot(grid[0, 1])
    ax_ga19b = fig.add_subplot(grid[1, 0])
    ax_ga19c = fig.add_subplot(grid[1, 1])

    ga18b_image = load_image_array(EXP1_DIR / "ga18b_baseline_default_seed42.png")
    ga18c_image = load_image_array(EXP1_DIR / "ga18c_baseline_default_seed42.png")
    if ga18b_image is None:
        add_missing_panel(ax_ga18b)
    else:
        ax_ga18b.imshow(ga18b_image)
        ax_ga18b.set_xticks([])
        ax_ga18b.set_yticks([])
    if ga18c_image is None:
        add_missing_panel(ax_ga18c)
    else:
        ax_ga18c.imshow(ga18c_image)
        ax_ga18c.set_xticks([])
        ax_ga18c.set_yticks([])

    ax_ga18b.set_title("GA18B baseline")
    ax_ga18c.set_title("GA18C baseline")

    ax_ga19b.axis("off")
    if ga19b_row is None:
        ax_ga19b.text(0.5, 0.5, "Missing GA19B baseline row", ha="center", va="center")
    else:
        ground_truth = textwrap.fill(ga19b_row.get("input_text", ""), width=45)
        prediction = textwrap.fill(ga19b_row.get("output_path", "") or ga19b_row.get("reference_image", ""), width=45)
        predicted_transcription = textwrap.fill(ga19b_row.get("prompt", ""), width=45)
        display_text = (
            "Ground truth transcription\n\n"
            f"{ground_truth}\n\n"
            "Predicted transcription\n\n"
            f"{textwrap.fill(ga19b_row.get('model', ''), width=45)}\n"
            f"{textwrap.fill(ga19b_row.get('reference_image', ''), width=45)}\n"
        )
        if ga19b_row.get("input_text") and ga19b_row.get("output_path"):
            display_text = (
                "Ground truth transcription\n\n"
                f"{ground_truth}\n\n"
                "Predicted transcription\n\n"
                f"{textwrap.fill(ga19b_row['output_path'], width=45)}"
            )
        ax_ga19b.text(0.02, 0.98, display_text, ha="left", va="top", fontsize=9)
    ax_ga19b.set_title("GA19B baseline transcription")

    baseline_audio = EXP1_DIR / "ga19c_baseline_default_seed42.wav"
    if baseline_audio.exists():
        sampling_rate, audio = read_audio_mono(baseline_audio)
        time_axis = np.arange(audio.size) / float(sampling_rate)
        ax_ga19c.plot(time_axis, audio, color=STYLE["colors"][0], linewidth=0.8)
        ax_ga19c.set_xlabel("Time (seconds)")
        ax_ga19c.set_ylabel("Amplitude")
        ax_ga19c.grid(True, alpha=STYLE["grid_alpha"])
    else:
        add_missing_panel(ax_ga19c)
    ax_ga19c.set_title("GA19C baseline waveform")

    plt.tight_layout()
    save_figure(fig, EXP1_DIR / "exp1_baseline_gallery.png", generated_paths)


def figure_02_ga18b_scale_sweep(generated_paths):
    scales = [round(step / 10.0, 2) for step in range(11)]
    seeds = [42, 123, 456]
    image_paths = [
        [EXP2_DIR / f"ga18b_2a_scale_{scale:.2f}_seed{seed}.png" for seed in seeds]
        for scale in scales
    ]
    create_image_grid(
        image_paths,
        [f"scale={scale:.1f}" for scale in scales],
        [f"seed={seed}" for seed in seeds],
        "Experiment 2A: GA18B Uniform Scale Sweep",
        EXP2_DIR / "ga18b_scale_sweep_grid.png",
        generated_paths,
    )


def figure_03_ga18b_reference_comparison(generated_paths):
    scales = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    seeds = [42, 123]
    image_paths = [
        [EXP2_DIR / f"ga18b_2b_scale_{scale:.2f}_seed{seed}.png" for seed in seeds]
        for scale in scales
    ]
    create_image_grid(
        image_paths,
        [f"scale={scale:.1f}" for scale in scales],
        [f"seed={seed}" for seed in seeds],
        "Experiment 2B: GA18B Alternative Reference Comparison",
        EXP2_DIR / "ga18b_reference_comparison.png",
        generated_paths,
    )


def figure_04_ga18b_steps_comparison(generated_paths):
    steps = [10, 25, 50]
    seeds = [42, 123]
    image_paths = [
        [EXP2_DIR / f"ga18b_2c_steps_{step}_seed{seed}.png" for seed in seeds]
        for step in steps
    ]
    create_image_grid(
        image_paths,
        [f"steps={step}" for step in steps],
        [f"seed={seed}" for seed in seeds],
        "Experiment 2C: GA18B Inference Step Comparison",
        EXP2_DIR / "ga18b_steps_comparison.png",
        generated_paths,
    )


def figure_05_ga18b_timing_vs_scale(generated_paths):
    rows = [row for row in read_csv_rows(EXP2_DIR / "exp2_ga18b_scale.csv") if row.get("sub_experiment") == "2a"]
    timing_by_scale = defaultdict(list)
    for row in rows:
        timing_by_scale[to_float(row.get("ip_adapter_scale"))].append(to_float(row.get("generation_time_sec")))
    x_values = sorted(timing_by_scale)
    y_values = [compute_mean(timing_by_scale[x_value]) for x_value in x_values]
    create_line_series_plot(
        [("GA18B mean time", x_values, y_values)],
        "Experiment 2: GA18B Generation Time vs. Uniform Scale",
        "IP-Adapter scale",
        "Mean generation time (seconds)",
        EXP2_DIR / "ga18b_timing_vs_scale.png",
        generated_paths,
    )


def figure_06_ga18c_layer_variation(generated_paths):
    layer_tags = ["001", "010", "011", "100", "110", "111"]
    seeds = [42, 123, 456]
    image_paths = [
        [EXP3_DIR / f"ga18c_3a_layer_{layer_tag}_seed{seed}.png" for seed in seeds]
        for layer_tag in layer_tags
    ]
    create_image_grid(
        image_paths,
        [f"layers={layer_tag}" for layer_tag in layer_tags],
        [f"seed={seed}" for seed in seeds],
        "Experiment 3A: GA18C Layer Activation Patterns",
        EXP3_DIR / "ga18c_layer_variation.png",
        generated_paths,
    )


def figure_07_ga18c_block_comparison(generated_paths):
    blocks = ["block0", "block1", "block2"]
    seeds = [42, 123]
    image_paths = [
        [EXP3_DIR / f"ga18c_3b_{block}_seed{seed}.png" for seed in seeds]
        for block in blocks
    ]
    create_image_grid(
        image_paths,
        [block for block in blocks],
        [f"seed={seed}" for seed in seeds],
        "Experiment 3B: GA18C Upsampling Block Comparison",
        EXP3_DIR / "ga18c_block_comparison.png",
        generated_paths,
    )


def figure_08_ga18c_scale_intensity(generated_paths):
    scales = [0.25, 0.50, 0.75, 1.00, 1.50]
    seeds = [42, 123]
    image_paths = [
        [EXP3_DIR / f"ga18c_3c_scale_{scale:.2f}_seed{seed}.png" for seed in seeds]
        for scale in scales
    ]
    create_image_grid(
        image_paths,
        [f"active={scale:.2f}" for scale in scales],
        [f"seed={seed}" for seed in seeds],
        "Experiment 3C: GA18C Scale Intensity Sweep",
        EXP3_DIR / "ga18c_scale_intensity.png",
        generated_paths,
    )


def figure_09_ga18c_prompt_comparison(generated_paths):
    prompt_tags = ["cat", "cityscape", "mountain", "portrait"]
    seeds = [42, 123]
    image_paths = [
        [EXP3_DIR / f"ga18c_3d_prompt_{prompt_tag}_seed{seed}.png" for seed in seeds]
        for prompt_tag in prompt_tags
    ]
    create_image_grid(
        image_paths,
        prompt_tags,
        [f"seed={seed}" for seed in seeds],
        "Experiment 3D: GA18C Prompt Comparison",
        EXP3_DIR / "ga18c_prompt_comparison.png",
        generated_paths,
    )


def figure_10_ga18c_reference_comparison(generated_paths):
    references = ["mamoeiro", "items_variation"]
    seeds = [42, 123]
    image_paths = [
        [EXP3_DIR / f"ga18c_3e_ref_{reference}_seed{seed}.png" for seed in seeds]
        for reference in references
    ]
    create_image_grid(
        image_paths,
        references,
        [f"seed={seed}" for seed in seeds],
        "Experiment 3E: GA18C Reference Image Comparison",
        EXP3_DIR / "ga18c_reference_comparison.png",
        generated_paths,
    )


def figure_11_ga18b_text_image_interaction(generated_paths):
    scales = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    seeds = [42, 123]
    image_paths = [
        [EXP4_DIR / f"ga18b_4a_scale_{scale:.2f}_seed{seed}.png" for seed in seeds]
        for scale in scales
    ]
    create_image_grid(
        image_paths,
        [f"scale={scale:.1f}" for scale in scales],
        [f"seed={seed}" for seed in seeds],
        "Experiment 4A: GA18B Text and Image Interaction",
        EXP4_DIR / "ga18b_text_image_interaction.png",
        generated_paths,
    )


def figure_12_ga18b_prompt_specificity(generated_paths):
    prompt_types = ["minimal", "moderate", "detailed"]
    seeds = [42, 123]
    image_paths = [
        [EXP4_DIR / f"ga18b_4b_prompt_{prompt_type}_seed{seed}.png" for seed in seeds]
        for prompt_type in prompt_types
    ]
    create_image_grid(
        image_paths,
        prompt_types,
        [f"seed={seed}" for seed in seeds],
        "Experiment 4B: GA18B Prompt Specificity",
        EXP4_DIR / "ga18b_prompt_specificity.png",
        generated_paths,
    )


def figure_13_ga18b_semantic_compatibility(generated_paths):
    prompt_types = ["compatible", "neutral", "incompatible"]
    seeds = [42, 123]
    image_paths = [
        [EXP4_DIR / f"ga18b_4c_prompt_{prompt_type}_seed{seed}.png" for seed in seeds]
        for prompt_type in prompt_types
    ]
    create_image_grid(
        image_paths,
        prompt_types,
        [f"seed={seed}" for seed in seeds],
        "Experiment 4C: GA18B Semantic Compatibility",
        EXP4_DIR / "ga18b_semantic_compatibility.png",
        generated_paths,
    )


def figure_14_uniform_vs_block(generated_paths):
    seeds = [42, 123, 456]
    image_paths = [
        [EXP4_DIR / f"ga18b_4e_mode_uniform_seed{seed}.png" for seed in seeds],
        [EXP4_DIR / f"ga18c_4e_mode_block_seed{seed}.png" for seed in seeds],
    ]
    create_image_grid(
        image_paths,
        ["uniform", "block"],
        [f"seed={seed}" for seed in seeds],
        "Experiment 4E: Uniform vs. Block-Level Conditioning",
        EXP4_DIR / "uniform_vs_block_comparison.png",
        generated_paths,
    )


def figure_15_asr_model_comparison(generated_paths):
    rows = [row for row in read_csv_rows(EXP5_DIR / "exp5_ga19b_asr.csv") if row.get("sub_experiment") == "5a"]
    metrics = defaultdict(list)
    for row in rows:
        metrics[row.get("model_name", "unknown")].append(to_float(row.get("wer")))
    ordered = sorted(((model, compute_mean(values)) for model, values in metrics.items()), key=lambda item: item[1])
    create_bar_chart(
        [item[0] for item in ordered],
        [item[1] for item in ordered],
        "Experiment 5A: Mean WER by ASR Model",
        "Model",
        "Mean WER",
        EXP5_DIR / "asr_model_comparison_bar.png",
        generated_paths,
    )


def figure_16_asr_wer_distribution(generated_paths):
    rows = [row for row in read_csv_rows(EXP5_DIR / "exp5_ga19b_asr.csv") if row.get("sub_experiment") == "5a"]
    data_map = defaultdict(list)
    for row in rows:
        data_map[row.get("model_name", "unknown")].append(to_float(row.get("wer")))
    create_box_plot(
        dict(sorted(data_map.items())),
        "Experiment 5A: WER Distribution by ASR Model",
        "Model",
        "WER",
        EXP5_DIR / "asr_wer_distribution.png",
        generated_paths,
    )


def figure_17_asr_sampling_rate_effect(generated_paths):
    rows = [row for row in read_csv_rows(EXP5_DIR / "exp5_ga19b_asr.csv") if row.get("sub_experiment") == "5b"]
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        model_name = row.get("model_name", "unknown")
        sampling_rate = int(to_float(row.get("sampling_rate"), default=0))
        grouped[model_name][sampling_rate].append(to_float(row.get("wer")))
    series_map = []
    for model_name, values_by_rate in sorted(grouped.items()):
        x_values = sorted(values_by_rate)
        y_values = [compute_mean(values_by_rate[x_value]) for x_value in x_values]
        series_map.append((model_name, x_values, y_values))
    create_line_series_plot(
        series_map,
        "Experiment 5B: WER vs. Sampling Rate",
        "Sampling rate (Hz)",
        "Mean WER",
        EXP5_DIR / "asr_sampling_rate_effect.png",
        generated_paths,
    )


def figure_18_asr_cer_comparison(generated_paths):
    rows = [row for row in read_csv_rows(EXP5_DIR / "exp5_ga19b_asr.csv") if row.get("sub_experiment") == "5a"]
    metrics = defaultdict(list)
    for row in rows:
        metrics[row.get("model_name", "unknown")].append(to_float(row.get("cer")))
    ordered = sorted(((model, compute_mean(values)) for model, values in metrics.items()), key=lambda item: item[1])
    create_bar_chart(
        [item[0] for item in ordered],
        [item[1] for item in ordered],
        "Experiment 5A: Mean CER by ASR Model",
        "Model",
        "Mean CER",
        EXP5_DIR / "asr_cer_model_comparison.png",
        generated_paths,
    )


def figure_19_asr_inference_time(generated_paths):
    rows = [row for row in read_csv_rows(EXP5_DIR / "exp5_ga19b_asr.csv") if row.get("sub_experiment") == "5a"]
    metrics = defaultdict(list)
    for row in rows:
        metrics[row.get("model_name", "unknown")].append(to_float(row.get("inference_time_sec")))
    ordered = sorted(((model, compute_mean(values)) for model, values in metrics.items()), key=lambda item: item[1])
    create_bar_chart(
        [item[0] for item in ordered],
        [item[1] for item in ordered],
        "Experiment 5A: Mean Inference Time by ASR Model",
        "Model",
        "Mean inference time (seconds)",
        EXP5_DIR / "asr_inference_time.png",
        generated_paths,
    )


def figure_20_tts_consistency_waveforms(generated_paths):
    seeds = [42, 123, 456, 789, 1024]
    audio_paths = [EXP6_DIR / f"ga19c_6a_repeat_seed{seed}.wav" for seed in seeds]
    create_waveform_comparison(
        audio_paths,
        [f"seed={seed}" for seed in seeds],
        "Experiment 6A: Bark Repeated-Generation Waveform Comparison",
        EXP6_DIR / "tts_consistency_waveforms.png",
        generated_paths,
    )


def figure_21_tts_text_length_analysis(generated_paths):
    rows = [row for row in read_csv_rows(EXP6_DIR / "exp6_ga19c_tts.csv") if row.get("sub_experiment") == "6b"]
    categories = ["short", "medium", "long"]
    duration_map = defaultdict(list)
    timing_map = defaultdict(list)
    for row in rows:
        category = row.get("text_category", "unknown")
        duration_map[category].append(to_float(row.get("audio_duration_sec")))
        timing_map[category].append(to_float(row.get("generation_time_sec")))
    create_grouped_bar_chart(
        categories,
        {
            "Audio duration (s)": [compute_mean(duration_map[category]) for category in categories],
            "Generation time (s)": [compute_mean(timing_map[category]) for category in categories],
        },
        "Experiment 6B: Text Length vs. Audio Duration and Generation Time",
        "Text length category",
        "Seconds",
        EXP6_DIR / "tts_text_length_analysis.png",
        generated_paths,
    )


def figure_22_tts_round_trip_wer(generated_paths):
    rows = [row for row in read_csv_rows(EXP6_DIR / "exp6_ga19c_tts.csv") if row.get("round_trip_wer") not in ("", None)]
    categories = [row.get("text_category", "unknown") for row in rows]
    values = [to_float(row.get("round_trip_wer")) for row in rows]
    create_bar_chart(
        categories,
        values,
        "Experiment 6D: Round-Trip WER by Text Category",
        "Text category",
        "Round-trip WER",
        EXP6_DIR / "tts_round_trip_wer.png",
        generated_paths,
    )


def figure_23_tts_generation_time_scaling(generated_paths):
    rows = [
        row
        for row in read_csv_rows(EXP6_DIR / "exp6_ga19c_tts.csv")
        if row.get("sub_experiment") in {"6b", "6c"}
    ]
    points_by_group = defaultdict(list)
    for row in rows:
        points_by_group[row.get("text_category", "unknown")].append(
            (to_float(row.get("text_length_words")), to_float(row.get("generation_time_sec")))
        )
    create_scatter_plot(
        dict(sorted(points_by_group.items())),
        "Experiments 6B and 6C: Generation Time vs. Word Count",
        "Input length (words)",
        "Generation time (seconds)",
        EXP6_DIR / "tts_generation_time_scaling.png",
        generated_paths,
    )


def figure_24_seed_sensitivity_ga18b(generated_paths):
    seeds = [42, 123, 456, 789, 1024, 2048, 3000, 4096]
    image_paths = [EXP7_DIR / f"ga18b_seed_seed{seed}.png" for seed in seeds]
    create_gallery(
        image_paths,
        [f"seed={seed}" for seed in seeds],
        "Experiment 7: GA18B Seed Sensitivity",
        EXP7_DIR / "seed_sensitivity_ga18b.png",
        generated_paths,
        ncols=4,
    )


def figure_25_seed_sensitivity_ga18c(generated_paths):
    seeds = [42, 123, 456, 789, 1024, 2048, 3000, 4096]
    image_paths = [EXP7_DIR / f"ga18c_seed_seed{seed}.png" for seed in seeds]
    create_gallery(
        image_paths,
        [f"seed={seed}" for seed in seeds],
        "Experiment 7: GA18C Seed Sensitivity",
        EXP7_DIR / "seed_sensitivity_ga18c.png",
        generated_paths,
        ncols=4,
    )


def figure_26_seed_sensitivity_ga19c(generated_paths):
    rows = [row for row in read_csv_rows(EXP7_DIR / "exp7_seed_sensitivity.csv") if row.get("script") == "GA19C"]
    seeds = [42, 123, 456, 789, 1024, 2048, 3000, 4096]
    row_by_seed = {int(to_float(row.get("seed"), default=-1)): row for row in rows}
    audio_paths = [EXP7_DIR / f"ga19c_seed_default_seed{seed}.wav" for seed in seeds]
    generation_times = [to_float(row_by_seed[seed].get("generation_time_sec")) for seed in seeds]
    create_tts_seed_summary(
        audio_paths,
        [f"seed={seed}" for seed in seeds],
        generation_times,
        EXP7_DIR / "seed_sensitivity_ga19c.png",
        generated_paths,
    )


def figure_27_seed_sensitivity_summary(generated_paths):
    rows = [row for row in read_csv_rows(EXP7_DIR / "exp7_seed_sensitivity.csv") if row.get("script") == "GA19C"]
    generation_time_by_seed = {
        int(to_float(row.get("seed"), default=-1)): to_float(row.get("generation_time_sec")) for row in rows
    }
    seeds = [42, 123, 456, 789, 1024, 2048, 3000, 4096]

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("Experiment 7: Seed Sensitivity Summary Across Paradigms", fontsize=STYLE["title_size"], fontweight="bold", y=1.01)
    grid = fig.add_gridspec(3, 8, height_ratios=[1.0, 1.0, 0.9])

    for idx, seed in enumerate(seeds):
        ax = fig.add_subplot(grid[0, idx])
        image = load_image_array(EXP7_DIR / f"ga18b_seed_seed{seed}.png")
        if image is None:
            add_missing_panel(ax)
        else:
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(str(seed), fontsize=STYLE["tick_size"])
        if idx == 0:
            ax.set_ylabel("GA18B", rotation=0, labelpad=35, va="center")

    for idx, seed in enumerate(seeds):
        ax = fig.add_subplot(grid[1, idx])
        image = load_image_array(EXP7_DIR / f"ga18c_seed_seed{seed}.png")
        if image is None:
            add_missing_panel(ax)
        else:
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
        if idx == 0:
            ax.set_ylabel("GA18C", rotation=0, labelpad=35, va="center")

    ax_bar = fig.add_subplot(grid[2, :])
    times = [generation_time_by_seed[seed] for seed in seeds]
    ax_bar.bar([str(seed) for seed in seeds], times, color=STYLE["colors"][2], edgecolor="black", alpha=0.85)
    ax_bar.set_xlabel("Seed")
    ax_bar.set_ylabel("Generation time (seconds)")
    ax_bar.set_title("GA19C generation time by seed")
    ax_bar.grid(axis="y", alpha=STYLE["grid_alpha"])

    plt.tight_layout()
    save_figure(fig, EXP7_DIR / "seed_sensitivity_summary.png", generated_paths)


def collect_timing_samples():
    exp1_rows = read_csv_rows(EXP1_DIR / "exp1_baselines.csv")
    exp2_rows = read_csv_rows(EXP2_DIR / "exp2_ga18b_scale.csv")
    exp3_rows = read_csv_rows(EXP3_DIR / "exp3_ga18c_blocks.csv")
    exp4_rows = read_csv_rows(EXP4_DIR / "exp4_text_image_interaction.csv")
    exp5_rows = read_csv_rows(EXP5_DIR / "exp5_ga19b_asr.csv")
    exp6_rows = read_csv_rows(EXP6_DIR / "exp6_ga19c_tts.csv")
    exp7_rows = read_csv_rows(EXP7_DIR / "exp7_seed_sensitivity.csv")

    ga18b_times = [to_float(row.get("generation_time_sec")) for row in exp1_rows if row.get("script") == "GA18B"]
    ga18b_times += [to_float(row.get("generation_time_sec")) for row in exp2_rows]
    ga18b_times += [to_float(row.get("generation_time_sec")) for row in exp4_rows if row.get("script") == "GA18B"]
    ga18b_times += [to_float(row.get("generation_time_sec")) for row in exp7_rows if row.get("script") == "GA18B"]

    ga18c_times = [to_float(row.get("generation_time_sec")) for row in exp1_rows if row.get("script") == "GA18C"]
    ga18c_times += [to_float(row.get("generation_time_sec")) for row in exp3_rows]
    ga18c_times += [to_float(row.get("generation_time_sec")) for row in exp4_rows if row.get("script") == "GA18C"]
    ga18c_times += [to_float(row.get("generation_time_sec")) for row in exp7_rows if row.get("script") == "GA18C"]

    ga19b_times = [to_float(row.get("generation_time_sec")) for row in exp1_rows if row.get("script") == "GA19B"]
    ga19b_times += [to_float(row.get("inference_time_sec")) for row in exp5_rows]

    ga19c_times = [to_float(row.get("generation_time_sec")) for row in exp1_rows if row.get("script") == "GA19C"]
    ga19c_times += [to_float(row.get("generation_time_sec")) for row in exp6_rows]
    ga19c_times += [to_float(row.get("generation_time_sec")) for row in exp7_rows if row.get("script") == "GA19C"]

    return {
        "GA18B uniform": [value for value in ga18b_times if not np.isnan(value)],
        "GA18C block": [value for value in ga18c_times if not np.isnan(value)],
        "GA19B ASR": [value for value in ga19b_times if not np.isnan(value)],
        "GA19C Bark": [value for value in ga19c_times if not np.isnan(value)],
    }


def collect_timing_summary():
    return {
        paradigm: compute_mean(values)
        for paradigm, values in collect_timing_samples().items()
    }


def get_cross_modal_comparison_records():
    timing_summary = collect_timing_summary()
    return [
        {
            "paradigm": "GA18B uniform",
            "output_type": "image",
            "conditioning_signal_type": "CLIP image\nembedding",
            "controllable_parameters_count": 5,
            "controllable_parameters": [
                "reference image",
                "text prompt",
                "IP-Adapter scale",
                "inference steps",
                "seed",
            ],
            "control_summary": "5: ref image,\nprompt, scale,\nsteps, seed",
            "stochasticity": "stochastic",
            "evaluation_metrics": "qualitative\nonly",
            "computational_cost": categorize_computational_cost(timing_summary["GA18B uniform"]),
            "mean_time_seconds": timing_summary["GA18B uniform"],
        },
        {
            "paradigm": "GA18C block",
            "output_type": "image",
            "conditioning_signal_type": "image embedding\n+ text tokens",
            "controllable_parameters_count": 6,
            "controllable_parameters": [
                "reference image",
                "text prompt",
                "block config",
                "active scale",
                "inference steps",
                "seed",
            ],
            "control_summary": "6: ref image,\nprompt, block cfg,\nscale, steps, seed",
            "stochasticity": "stochastic",
            "evaluation_metrics": "qualitative\nonly",
            "computational_cost": categorize_computational_cost(timing_summary["GA18C block"]),
            "mean_time_seconds": timing_summary["GA18C block"],
        },
        {
            "paradigm": "GA19B ASR",
            "output_type": "text\ntranscription",
            "conditioning_signal_type": "audio waveform",
            "controllable_parameters_count": 3,
            "controllable_parameters": [
                "ASR model",
                "input audio",
                "sampling rate",
            ],
            "control_summary": "3: model,\naudio, sample rate",
            "stochasticity": "deterministic",
            "evaluation_metrics": "WER/CER\n+ inference time",
            "computational_cost": categorize_computational_cost(timing_summary["GA19B ASR"]),
            "mean_time_seconds": timing_summary["GA19B ASR"],
        },
        {
            "paradigm": "GA19C Bark",
            "output_type": "audio\nwaveform",
            "conditioning_signal_type": "text tokens",
            "controllable_parameters_count": 2,
            "controllable_parameters": [
                "input text",
                "seed",
            ],
            "control_summary": "2: input text,\nseed",
            "stochasticity": "stochastic",
            "evaluation_metrics": "generation time\n+ round-trip WER",
            "computational_cost": categorize_computational_cost(timing_summary["GA19C Bark"]),
            "mean_time_seconds": timing_summary["GA19C Bark"],
        },
    ]


def figure_28_cross_modal_comparison_table(generated_paths):
    comparison_records = get_cross_modal_comparison_records()
    rows = [
        [
            record["paradigm"],
            record["output_type"],
            record["conditioning_signal_type"],
            record["control_summary"],
            record["stochasticity"],
            record["evaluation_metrics"],
            record["computational_cost"],
            f"{record['mean_time_seconds']:.2f}",
        ]
        for record in comparison_records
    ]
    create_table_figure(
        [
            "Paradigm",
            "Output",
            "Conditioning\nsignal",
            "Controllable\nparams",
            "Stochasticity",
            "Evaluation\nmetrics",
            "Relative\ncost",
            "Mean time\n(s)",
        ],
        rows,
        "Cross-Modal Paradigm Comparison",
        EXP_DIR / "cross_modal_comparison_table.png",
        generated_paths,
        figsize=(18, 5.6),
        font_size=8,
        y_scale=1.95,
    )


def figure_29_cross_timing_comparison(generated_paths):
    timing_summary = collect_timing_summary()
    create_bar_chart(
        list(timing_summary.keys()),
        list(timing_summary.values()),
        "Cross-Paradigm Timing Comparison",
        "Paradigm",
        "Average time per sample (seconds)",
        EXP_DIR / "cross_timing_comparison.png",
        generated_paths,
    )


def generate_all_figures():
    ensure_directories()
    generated_paths: list[Path] = []

    figure_01_baseline_gallery(generated_paths)
    figure_02_ga18b_scale_sweep(generated_paths)
    figure_03_ga18b_reference_comparison(generated_paths)
    figure_04_ga18b_steps_comparison(generated_paths)
    figure_05_ga18b_timing_vs_scale(generated_paths)
    figure_06_ga18c_layer_variation(generated_paths)
    figure_07_ga18c_block_comparison(generated_paths)
    figure_08_ga18c_scale_intensity(generated_paths)
    figure_09_ga18c_prompt_comparison(generated_paths)
    figure_10_ga18c_reference_comparison(generated_paths)
    figure_11_ga18b_text_image_interaction(generated_paths)
    figure_12_ga18b_prompt_specificity(generated_paths)
    figure_13_ga18b_semantic_compatibility(generated_paths)
    figure_14_uniform_vs_block(generated_paths)
    figure_15_asr_model_comparison(generated_paths)
    figure_16_asr_wer_distribution(generated_paths)
    figure_17_asr_sampling_rate_effect(generated_paths)
    figure_18_asr_cer_comparison(generated_paths)
    figure_19_asr_inference_time(generated_paths)
    figure_20_tts_consistency_waveforms(generated_paths)
    figure_21_tts_text_length_analysis(generated_paths)
    figure_22_tts_round_trip_wer(generated_paths)
    figure_23_tts_generation_time_scaling(generated_paths)
    figure_24_seed_sensitivity_ga18b(generated_paths)
    figure_25_seed_sensitivity_ga18c(generated_paths)
    figure_26_seed_sensitivity_ga19c(generated_paths)
    figure_27_seed_sensitivity_summary(generated_paths)
    figure_28_cross_modal_comparison_table(generated_paths)
    figure_29_cross_timing_comparison(generated_paths)

    copy_report_figures(generated_paths)
    return generated_paths


def main():
    generated_paths = generate_all_figures()
    print(f"[FIGURE] Generated {len(generated_paths)} figure(s).")


if __name__ == "__main__":
    main()