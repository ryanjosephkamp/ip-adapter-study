from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = PROJECT_ROOT / "hw13_experiments"
NB_DIR = PROJECT_ROOT / "hw13_scripts" / "notebooks"


def main():
    assert (NB_DIR / "group_a_dry_run.ipynb").exists(), "FAIL: Dry-run notebook missing"
    assert (NB_DIR / "group_a_full.ipynb").exists(), "FAIL: Full notebook missing"
    print("TEST 0 PASS: Both Group A notebooks exist.")

    ga18b_base = list((EXP_DIR / "exp1_baselines").glob("ga18b_*.png"))
    ga18c_base = list((EXP_DIR / "exp1_baselines").glob("ga18c_*.png"))
    assert len(ga18b_base) >= 1, f"FAIL: Expected >=1 GA18B baseline, found {len(ga18b_base)}"
    assert len(ga18c_base) >= 1, f"FAIL: Expected >=1 GA18C baseline, found {len(ga18c_base)}"
    print(f"TEST 1 PASS: {len(ga18b_base)} GA18B + {len(ga18c_base)} GA18C baseline image(s).")

    exp2_files = list((EXP_DIR / "exp2_ga18b_scale").glob("ga18b_*.png"))
    assert len(exp2_files) >= 45, f"FAIL: Expected >=45 Exp 2 images, found {len(exp2_files)}"
    print(f"TEST 2 PASS: {len(exp2_files)} Exp 2 images.")

    exp3_files = list((EXP_DIR / "exp3_ga18c_blocks").glob("ga18c_*.png"))
    assert len(exp3_files) >= 40, f"FAIL: Expected >=40 Exp 3 images, found {len(exp3_files)}"
    print(f"TEST 3 PASS: {len(exp3_files)} Exp 3 images.")

    exp4_files = list((EXP_DIR / "exp4_text_image_interaction").glob("ga18*.png"))
    assert len(exp4_files) >= 28, f"FAIL: Expected >=28 Exp 4 images, found {len(exp4_files)}"
    print(f"TEST 4 PASS: {len(exp4_files)} Exp 4 images.")

    seed_ga18b = list((EXP_DIR / "exp7_seed_sensitivity").glob("ga18b_*.png"))
    seed_ga18c = list((EXP_DIR / "exp7_seed_sensitivity").glob("ga18c_*.png"))
    assert len(seed_ga18b) >= 8, f"FAIL: Expected >=8 GA18B seed images, found {len(seed_ga18b)}"
    assert len(seed_ga18c) >= 8, f"FAIL: Expected >=8 GA18C seed images, found {len(seed_ga18c)}"
    print(f"TEST 5 PASS: {len(seed_ga18b)} GA18B + {len(seed_ga18c)} GA18C seed sensitivity images.")

    for exp_name, min_rows in [
        ("exp2_ga18b_scale", 45),
        ("exp3_ga18c_blocks", 40),
        ("exp4_text_image_interaction", 28),
    ]:
        csv_path = EXP_DIR / exp_name / f"{exp_name}.csv"
        assert csv_path.exists(), f"FAIL: CSV missing: {csv_path}"
        with csv_path.open("r", encoding="utf-8") as handle:
            rows = sum(1 for _ in handle) - 1
        assert rows >= min_rows, f"FAIL: {exp_name} CSV has {rows} rows, expected >={min_rows}"
    print("TEST 6 PASS: All Group A CSVs exist with correct row counts.")

    all_images = exp2_files + exp3_files + exp4_files + seed_ga18b + seed_ga18c
    for image_path in all_images:
        assert image_path.stat().st_size > 1000, (
            f"FAIL: Suspiciously small image: {image_path} ({image_path.stat().st_size} bytes)"
        )
    print(f"TEST 7 PASS: All {len(all_images)} images are non-empty.")

    print("\nSTEP 3 VERIFICATION: ALL TESTS PASSED")


if __name__ == "__main__":
    main()