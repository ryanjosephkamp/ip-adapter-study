import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = PROJECT_ROOT / "hw13_experiments"
REPORT_FIGS = PROJECT_ROOT / "hw13_reports" / "figures"


def main():
    summary_path = EXP_DIR / "summary_statistics.json"
    assert summary_path.exists(), "FAIL: summary_statistics.json missing"
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    assert "total_images_generated" in summary
    assert "total_audio_files_generated" in summary
    assert "total_transcriptions" in summary
    assert "total_gpu_time_sec" in summary
    assert "per_paradigm_timing" in summary
    assert "experiment_completion" in summary
    print(
        "TEST 1 PASS: Summary statistics exist. "
        f"Images: {summary['total_images_generated']}, "
        f"Audio: {summary['total_audio_files_generated']}, "
        f"Transcriptions: {summary['total_transcriptions']}."
    )

    cross_figs = ["cross_timing_comparison.png", "cross_modal_comparison_table.png"]
    for fig_name in cross_figs:
        assert (REPORT_FIGS / fig_name).exists(), f"FAIL: Missing {fig_name}"
    print("TEST 2 PASS: Cross-paradigm figures exist.")

    total_images = summary.get("total_images_generated", 0)
    assert total_images >= 130, f"FAIL: Expected >=130 total images, found {total_images}"
    total_audio = summary.get("total_audio_files_generated", 0)
    assert total_audio >= 20, f"FAIL: Expected >=20 total audio files, found {total_audio}"
    print(f"TEST 3 PASS: {total_images} images, {total_audio} audio files - counts reasonable.")

    csv_names = [
        "exp1_baselines/exp1_baselines.csv",
        "exp2_ga18b_scale/exp2_ga18b_scale.csv",
        "exp3_ga18c_blocks/exp3_ga18c_blocks.csv",
        "exp4_text_image_interaction/exp4_text_image_interaction.csv",
        "exp5_ga19b_asr/exp5_ga19b_asr.csv",
        "exp6_ga19c_tts/exp6_ga19c_tts.csv",
        "exp7_seed_sensitivity/exp7_seed_sensitivity.csv",
    ]
    for csv_name in csv_names:
        assert (EXP_DIR / csv_name).exists(), f"FAIL: Missing CSV: {csv_name}"
    print("TEST 4 PASS: All 7 experiment CSVs exist.")

    print("\nSTEP 7 VERIFICATION: ALL TESTS PASSED")
    print("\nPHASE 2 IMPLEMENTATION COMPLETE - All experiments executed and analyzed.")


if __name__ == "__main__":
    main()