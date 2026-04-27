import argparse
import math
from pathlib import Path

import pandas as pd

from cskh_common import load_jsonl, write_jsonl

# Minimum rows needed to satisfy golden + dev allocation.
# Train gets whatever is left — no minimum enforced there.
_MIN_CLEAN = 480   # 160 (golden) + 320 (dev)
_MIN_REJECT = 120  # 40  (golden) + 80  (dev)


def _strip_nan(records: list[dict]) -> list[dict]:
    """Remove keys whose value is float NaN.

    pandas.DataFrame.to_dict('records') fills columns absent in some source
    rows with float('nan'). json.dumps silently serialises these as the bare
    literal ``NaN``, which is not valid JSON and corrupts every downstream
    JSONL loader. Strip those keys here so the written files are always valid.
    """
    return [
        {k: v for k, v in record.items()
         if not (isinstance(v, float) and math.isnan(v))}
        for record in records
    ]


def stratified_split(
    clean_rows: list[dict],
    reject_rows: list[dict],
    output_dir: Path,
    seed: int,
) -> dict[str, int]:
    n_clean = len(clean_rows)
    n_reject = len(reject_rows)

    if n_clean < _MIN_CLEAN:
        raise RuntimeError(
            f"Not enough clean rows: need {_MIN_CLEAN} for golden+dev, got {n_clean}. "
            "Check step-1 output for failures."
        )
    if n_reject < _MIN_REJECT:
        raise RuntimeError(
            f"Not enough reject rows: need {_MIN_REJECT} for golden+dev, got {n_reject}. "
            "Check step-2 output for failures."
        )

    # Warn when totals fall below original spec targets.
    if n_clean < 3_400:
        print(f"[WARN] Expected ~3400 clean rows, got {n_clean}. Train set will be smaller.")
    if n_reject < 800:
        print(f"[WARN] Expected ~800 reject rows, got {n_reject}. Train set will be smaller.")

    clean_df = (
        pd.DataFrame(clean_rows)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    reject_df = (
        pd.DataFrame(reject_rows)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )

    # Golden: 160 clean + 40 reject = 200 rows
    golden = _strip_nan(
        pd.concat([clean_df.iloc[:160], reject_df.iloc[:40]], ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .to_dict("records")
    )

    # Dev: 320 clean + 80 reject = 400 rows
    dev = _strip_nan(
        pd.concat([clean_df.iloc[160:480], reject_df.iloc[40:120]], ignore_index=True)
        .sample(frac=1.0, random_state=seed + 1)
        .to_dict("records")
    )

    # Train: all remaining rows
    train = _strip_nan(
        pd.concat([clean_df.iloc[480:], reject_df.iloc[120:]], ignore_index=True)
        .sample(frac=1.0, random_state=seed + 2)
        .to_dict("records")
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "golden_test_v2.jsonl", golden)
    write_jsonl(output_dir / "dev_v2.jsonl", dev)
    write_jsonl(output_dir / "train_v2.jsonl", train)

    return {"golden": len(golden), "dev": len(dev), "train": len(train)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3: Combine and stratified-split datasets.")
    parser.add_argument("--clean-input",  type=Path, default=Path("data/clean_old_data.jsonl"))
    parser.add_argument("--new-input",    type=Path, default=Path("data/new_reject_data.jsonl"))
    parser.add_argument("--output-dir",   type=Path, default=Path("data"))
    parser.add_argument("--seed",         type=int,  default=42)
    args = parser.parse_args()

    clean_rows  = load_jsonl(args.clean_input)
    reject_rows = load_jsonl(args.new_input)

    print(f"Loaded {len(clean_rows):,} clean rows and {len(reject_rows):,} synthetic reject rows.")

    stats = stratified_split(clean_rows, reject_rows, args.output_dir, args.seed)

    total = sum(stats.values())
    print("\n── Sanity check ──────────────────────────────────")
    for split, count in stats.items():
        pct = count / total * 100
        print(f"  {split:<12} {count:>5} rows  ({pct:.1f}%)")
    print(f"  {'TOTAL':<12} {total:>5} rows")

    expected_train = (len(clean_rows) - _MIN_CLEAN) + (len(reject_rows) - _MIN_REJECT)
    assert stats["golden"] == 200,            f"golden mismatch: {stats['golden']}"
    assert stats["dev"]    == 400,            f"dev mismatch: {stats['dev']}"
    assert stats["train"]  == expected_train, (
        f"train mismatch: {stats['train']} vs expected {expected_train}"
    )
    print("  ✓ All assertions passed.")


if __name__ == "__main__":
    main()
