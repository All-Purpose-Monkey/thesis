#!/usr/bin/env python3
"""
island_study.py  —  Contiguity ("island") analysis for UK-DALE binarized labels.

Purpose
-------
Quantify how fragmented the 6-second label grid is per house. Short, frequent
islands = little temporal contiguity = a shuffled iterative split mostly breaks
autocorrelation, so row-level evaluation/bootstrap is appropriate.

What it reports, per house and combined:
  - number of rows and number of contiguous islands
  - island length: median / mean / p90 / max  (in segments and minutes)
  - for block sizes 30s/1min/2min/5min: how many full blocks exist and what
    % of segments live in islands at least that long
  - lag-1 label persistence within contiguous pairs  P(on_{t+1} | on_t)
    (high persistence + tiny islands = adjacency is correlated but rare)

Usage
-----
  python island_study.py --input-dir /path/to/labels --pattern "*_binarized.csv"
  # or explicit files:
  python island_study.py --files house1_binarized.csv house2_binarized.csv ...

Outputs
-------
  - prints a readable report to stdout
  - writes island_summary.csv  (one row per house + a COMBINED row)

Assumptions
-----------
  - each CSV has a 'timestamp' column (UNIX seconds) on a fixed grid (default 6s,
    auto-detected as the median diff). Appliance columns are optional and only
    used for the persistence stat.
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd

# Block sizes to probe, as (label, segments-per-block). 6s grid => 5 seg = 30s.
BLOCKS = [("30s", 5), ("1min", 10), ("2min", 20), ("5min", 50)]
APP_COLS = ["kettle", "toaster", "microwave", "dishwasher", "fridge", "washing_machine"]


def island_sizes_from_timestamps(ts, step):
    """Return array of island lengths (in segments). An island is a run of rows
    whose consecutive timestamp difference equals `step`."""
    ts = np.sort(np.asarray(ts, dtype=np.int64))
    if len(ts) == 1:
        return np.array([1])
    d = np.diff(ts)
    # boundaries: positions where the gap is NOT exactly one step
    breaks = np.where(d != step)[0]
    bounds = np.concatenate(([-1], breaks, [len(ts) - 1]))
    return np.diff(bounds)  # lengths in segments


def lag1_persistence(df, ts, step):
    """P(on_{t+1} | on_t) computed only over contiguous (gap==step) pairs."""
    d = np.diff(np.asarray(ts, dtype=np.int64))
    contig = d == step
    out = {}
    for a in APP_COLS:
        if a not in df.columns:
            continue
        v = df[a].values.astype(np.int8)
        on = v[:-1] == 1
        m = contig & on
        out[a] = round(float((v[1:][m] == 1).mean()), 3) if m.sum() > 0 else float("nan")
    return out


def analyse_file(path, step_override=None):
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path}: no 'timestamp' column found")
    ts = df["timestamp"].values.astype(np.int64)
    order = np.argsort(ts)
    ts = ts[order]
    df = df.iloc[order].reset_index(drop=True)

    step = step_override if step_override else int(np.median(np.diff(ts))) if len(ts) > 1 else 6
    sizes = island_sizes_from_timestamps(ts, step)
    n = len(ts)
    secs = sizes * step

    row = {
        "house": os.path.basename(path),
        "step_s": step,
        "rows": n,
        "islands": len(sizes),
        "med_len_seg": int(np.median(sizes)),
        "mean_len_seg": round(float(sizes.mean()), 2),
        "p90_len_seg": int(np.percentile(sizes, 90)),
        "max_len_seg": int(sizes.max()),
        "med_len_min": round(float(np.median(secs)) / 60, 2),
        "max_len_min": round(float(secs.max()) / 60, 2),
    }
    for label, bs in BLOCKS:
        seg_in = int(sizes[sizes >= bs].sum())
        row[f"pct_in_islands_ge_{label}"] = round(100 * seg_in / n, 1)
        row[f"full_blocks_{label}"] = int((sizes // bs).sum())

    persist = lag1_persistence(df, ts, step)
    for a, p in persist.items():
        row[f"persist_{a}"] = p
    return row, sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=".", help="folder containing the label CSVs")
    ap.add_argument("--pattern", default="*_binarized.csv", help="glob pattern within input-dir")
    ap.add_argument("--files", nargs="*", help="explicit file paths (overrides --input-dir/--pattern)")
    ap.add_argument("--step", type=int, default=None, help="grid step in seconds (default: auto-detect)")
    ap.add_argument("--out", default="island_summary.csv", help="output summary CSV path")
    args = ap.parse_args()

    files = args.files if args.files else sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        raise SystemExit("No input files found. Check --input-dir / --pattern / --files.")

    rows, all_sizes = [], []
    for path in files:
        row, sizes = analyse_file(path, args.step)
        rows.append(row)
        all_sizes.append(sizes)
        print(f"\n=== {row['house']}  (step={row['step_s']}s) ===")
        print(f"  rows={row['rows']:,}  islands={row['islands']:,}")
        print(f"  island length: median={row['med_len_seg']} seg  mean={row['mean_len_seg']} seg  "
              f"p90={row['p90_len_seg']}  max={row['max_len_seg']} seg ({row['max_len_min']} min)")
        print("  block coverage (% of segments in islands >= size):")
        for label, _ in BLOCKS:
            print(f"    >= {label:5}: {row[f'pct_in_islands_ge_{label}']:>5}%   "
                  f"(full blocks: {row[f'full_blocks_{label}']:,})")
        pk = [k for k in row if k.startswith("persist_")]
        if pk:
            print("  lag-1 persistence P(on->on) within contiguous pairs:")
            print("    " + "  ".join(f"{k.replace('persist_',''):>14}={row[k]}" for k in pk))

    # COMBINED row across all houses
    combined_sizes = np.concatenate(all_sizes)
    n_tot = int(combined_sizes.sum())
    comb = {"house": "COMBINED", "step_s": rows[0]["step_s"], "rows": int(sum(r["rows"] for r in rows)),
            "islands": len(combined_sizes), "med_len_seg": int(np.median(combined_sizes)),
            "mean_len_seg": round(float(combined_sizes.mean()), 2),
            "p90_len_seg": int(np.percentile(combined_sizes, 90)),
            "max_len_seg": int(combined_sizes.max()), "med_len_min": "", "max_len_min": ""}
    for label, bs in BLOCKS:
        comb[f"pct_in_islands_ge_{label}"] = round(100 * int(combined_sizes[combined_sizes >= bs].sum()) / n_tot, 1)
        comb[f"full_blocks_{label}"] = int((combined_sizes // bs).sum())
    rows.append(comb)

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()