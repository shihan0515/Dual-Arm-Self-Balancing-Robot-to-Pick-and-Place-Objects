"""
Parse PPO evaluation log directory into an Excel file for thesis reporting.

Usage:
    python parse_eval_to_excel.py <ppo_log_dir> [output.xlsx]

Example:
    python parse_eval_to_excel.py eval_ppo_20260609_120000 eval_results.xlsx
"""

import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path


OBJECTS = ["mug", "drill", "dumbbell"]
SEEDS   = ["42", "7", "123"]

# Ordered easy → hard (Success Rate excluded)
METRICS = [
    ("p2r", "Phase2 Entry Rate"),
    ("tcr", "Tight Contact Rate"),
    ("psr", "Partial Success Rate"),
]

PATTERN = re.compile(
    r"final success_rate:\s*([0-9.]+).*?"
    r"final partial_success_rate:\s*([0-9.]+).*?"
    r"final phase2_rate:\s*([0-9.]+).*?"
    r"final tight_contact_rate:\s*([0-9.]+)"
)


def parse_log(path: Path) -> dict:
    last = None
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = PATTERN.search(line)
            if m:
                last = m
    if last:
        return {
            # "sr":  float(last.group(1)),   # excluded
            "psr": float(last.group(2)),
            "p2r": float(last.group(3)),
            "tcr": float(last.group(4)),
        }
    return {k: float("nan") for k, _ in METRICS}


def parse_dir(log_dir: str) -> dict:
    base = Path(log_dir)
    data = {}
    for obj in OBJECTS:
        data[obj] = {}
        for seed in SEEDS:
            log_file = base / f"{obj}_seed{seed}.log"
            if log_file.exists():
                data[obj][seed] = parse_log(log_file)
                print(f"  OK: {log_file.name}")
            else:
                data[obj][seed] = {k: float("nan") for k, _ in METRICS}
                print(f"  MISSING: {log_file.name}")
    return data


def build_sheet(ppo: dict, metric_key: str) -> pd.DataFrame:
    rows = []
    for obj in OBJECTS:
        row = {"Object": obj}
        vals = []
        for seed in SEEDS:
            v = ppo[obj][seed].get(metric_key, float("nan")) * 100
            row[f"seed_{seed}"] = round(v, 2)
            if not np.isnan(v):
                vals.append(v)
        row["Mean (%)"] = round(np.mean(vals), 2) if vals else float("nan")
        row["Std (%)"]  = round(np.std(vals),  2) if vals else float("nan")
        rows.append(row)

    # Average row
    avg = {"Object": "Average"}
    for col in rows[0]:
        if col == "Object":
            continue
        vals = [r[col] for r in rows if not np.isnan(r[col])]
        avg[col] = round(np.mean(vals), 2) if vals else float("nan")
    rows.append(avg)

    return pd.DataFrame(rows)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    ppo_dir  = sys.argv[1]
    out_xlsx = sys.argv[2] if len(sys.argv) > 2 else "eval_results.xlsx"

    print(f"\n[PPO] Parsing {ppo_dir}")
    ppo = parse_dir(ppo_dir)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        # One sheet per metric, ordered easy → hard
        for metric_key, metric_name in METRICS:
            df = build_sheet(ppo, metric_key)
            df.to_excel(writer, sheet_name=metric_name, index=False)
            print(f"  Sheet '{metric_name}' written.")

        # Summary sheet: all metrics in one table
        summary_rows = []
        for metric_key, metric_name in METRICS:
            df = build_sheet(ppo, metric_key)
            for _, row in df.iterrows():
                summary_rows.append({
                    "Metric":    metric_name,
                    "Object":    row["Object"],
                    "Mean (%)":  row["Mean (%)"],
                    "Std (%)":   row["Std (%)"],
                })
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
        print("  Sheet 'Summary' written.")

    print(f"\nSaved → {out_xlsx}")


if __name__ == "__main__":
    main()
