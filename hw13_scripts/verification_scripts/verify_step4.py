import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = PROJECT_ROOT / "hw13_experiments"
NB_DIR = PROJECT_ROOT / "hw13_scripts" / "notebooks"


def main():
    assert (NB_DIR / "group_b_dry_run.ipynb").exists(), "FAIL: Dry-run notebook missing"
    assert (NB_DIR / "group_b_full.ipynb").exists(), "FAIL: Full notebook missing"
    print("TEST 0 PASS: Both Group B notebooks exist.")

    exp5_csv = EXP_DIR / "exp5_ga19b_asr" / "exp5_ga19b_asr.csv"
    assert exp5_csv.exists(), "FAIL: Exp 5 CSV missing"
    with exp5_csv.open("r", encoding="utf-8") as handle:
        rows = sum(1 for _ in handle) - 1
    assert rows >= 140, f"FAIL: Expected >=140 CSV rows, found {rows}"
    print(f"TEST 1 PASS: Exp 5 CSV has {rows} data rows.")

    with exp5_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        valid_wer_count = 0
        for row in reader:
            wer = row.get("wer", "")
            cer = row.get("cer", "")
            if wer and cer and float(wer) >= 0 and float(cer) >= 0:
                valid_wer_count += 1
    assert valid_wer_count >= 140, f"FAIL: Only {valid_wer_count} rows have valid WER/CER"
    print(f"TEST 2 PASS: {valid_wer_count} rows with valid WER/CER values.")

    with exp5_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        models = {row.get("model_name", "") for row in reader if row.get("model_name", "")}
    assert len(models) >= 3, f"FAIL: Expected >=3 models, found {sorted(models)}"
    print(f"TEST 3 PASS: {len(models)} ASR models represented: {sorted(models)}")

    baseline_csv = EXP_DIR / "exp1_baselines" / "exp1_baselines.csv"
    assert baseline_csv.exists(), "FAIL: Baseline CSV missing"
    print("TEST 4 PASS: Baseline CSV exists.")

    print("\nSTEP 4 VERIFICATION: ALL TESTS PASSED")


if __name__ == "__main__":
    main()