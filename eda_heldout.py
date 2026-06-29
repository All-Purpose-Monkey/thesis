"""
eda_heldout.py — per-segment EDA on failure analysis CSVs

Usage:
    python eda_heldout.py <failure_analysis.csv>

Outputs three CSVs alongside the console print, named after the input file:
    <stem>_s0_complexity.csv     — complexity distribution + error rates
    <stem>_s1_performance.csv    — TP/FP/FN/TN/P/R/F1 by appliance × complexity
    <stem>_s2_confidence.csv     — error confidence breakdown by appliance × error type
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ─── CONFIG ─────────────────────────────────────────────────────────────
APPLIANCES = [
    "kettle", "toaster", "microwave",
    "dishwasher", "fridge", "washing_machine",
]

# |conf_diff| bands for classifying how wrong the model was
CONF_BANDS = [
    ("borderline", 0.00, 0.15),  # close to threshold — could go either way
    ("moderate",   0.15, 0.35),  # somewhat confident but wrong
    ("misfire",    0.35, 9.99),  # very confident and completely wrong
]
# ────────────────────────────────────────────────────────────────────────


def load(path):
    return pd.read_csv(path, index_col="timestamp")


def add_complexity_flag(df):
    true_cols = [f"{a}_true" for a in APPLIANCES]
    df["_n_active"] = df[true_cols].sum(axis=1).astype(int)
    df["complexity"] = df["_n_active"].map(
        lambda n: "negative" if n == 0 else ("simple" if n == 1 else "complex")
    )
    return df


def safe_f1(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def conf_band(abs_cd):
    for label, lo, hi in CONF_BANDS:
        if lo <= abs_cd < hi:
            return label
    return "misfire"


# ─── SECTION 0 ───────────────────────────────────────────────────────────

def build_s0(df):
    err_cols = [f"{a}_error" for a in APPLIANCES]
    total    = len(df)
    rows     = []
    for flag in ["negative", "simple", "complex"]:
        sub     = df[df["complexity"] == flag]
        n       = len(sub)
        pct     = 100 * n / total if total > 0 else 0.0
        any_err = sub[err_cols].isin(["FP", "FN"]).any(axis=1)
        err_n   = int(any_err.sum())
        err_pct = 100 * any_err.mean() if n > 0 else 0.0
        rows.append({
            "complexity":      flag,
            "n_segments":      n,
            "pct_of_total":    round(pct, 2),
            "n_with_any_error": err_n,
            "error_rate_pct":  round(err_pct, 2),
        })
    rows.append({
        "complexity":      "TOTAL",
        "n_segments":      total,
        "pct_of_total":    100.0,
        "n_with_any_error": int(df[err_cols].isin(["FP", "FN"]).any(axis=1).sum()),
        "error_rate_pct":  round(100 * df[err_cols].isin(["FP", "FN"]).any(axis=1).mean(), 2),
    })
    return pd.DataFrame(rows)


def section_overview(df, s0):
    sep()
    print("SECTION 0 — Segment Complexity Distribution")
    sep()
    print(s0.to_string(index=False))


# ─── SECTION 1 ───────────────────────────────────────────────────────────

def build_s1(df):
    flags = ["negative", "simple", "complex"]
    rows  = []
    for app in APPLIANCES:
        err_col = f"{app}_error"
        for flag in flags:
            sub    = df[df["complexity"] == flag]
            c      = sub[err_col].value_counts()
            tp, fp = c.get("TP", 0), c.get("FP", 0)
            fn, tn = c.get("FN", 0), c.get("TN", 0)
            prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            rows.append({
                "appliance":  app,
                "complexity": flag,
                "n_segments": len(sub),
                "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "precision": round(prec, 4),
                "recall":    round(rec, 4),
                "f1":        round(safe_f1(tp, fp, fn), 4),
            })
    return pd.DataFrame(rows)


def section_performance_by_flag(df, s1):
    sep()
    print("SECTION 1 — Performance by Appliance × Complexity")
    sep()
    hdr  = f"  {'Flag':<12} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>6}  {'P':>6} {'R':>6} {'F1':>6}  n"
    divr = f"  {'-'*12} {'-'*5} {'-'*5} {'-'*5} {'-'*6}  {'-'*6} {'-'*6} {'-'*6}  -"
    for app in APPLIANCES:
        print(f"\n  {app.upper()}")
        print(hdr)
        print(divr)
        sub = s1[s1["appliance"] == app]
        for _, r in sub.iterrows():
            print(f"  {r.complexity:<12} {r.TP:>5} {r.FP:>5} {r.FN:>5} {r.TN:>6}  "
                  f"{r.precision:>6.3f} {r.recall:>6.3f} {r.f1:>6.3f}  {r.n_segments}")


# ─── SECTION 2 ───────────────────────────────────────────────────────────

def build_s2(df):
    rows = []
    for app in APPLIANCES:
        err_col  = f"{app}_error"
        conf_col = f"{app}_conf_diff"
        for etype in ["FP", "FN"]:
            mask = df[err_col] == etype
            n    = int(mask.sum())
            if n == 0:
                continue
            sub_cd = df.loc[mask, conf_col]
            abs_cd = sub_cd.abs()
            bands  = abs_cd.apply(conf_band).value_counts()
            bl = int(bands.get("borderline", 0))
            md = int(bands.get("moderate",   0))
            mf = int(bands.get("misfire",    0))
            rows.append({
                "appliance":         app,
                "error_type":        etype,
                "n_errors":          n,
                "mean_conf_diff":    round(float(sub_cd.mean()), 4),
                "median_conf_diff":  round(float(sub_cd.median()), 4),
                "borderline_n":      bl,
                "borderline_pct":    round(100 * bl / n, 2),
                "moderate_n":        md,
                "moderate_pct":      round(100 * md / n, 2),
                "misfire_n":         mf,
                "misfire_pct":       round(100 * mf / n, 2),
            })
    return pd.DataFrame(rows)


def section_confidence(df, s2):
    sep()
    print("SECTION 2 — Error Confidence Breakdown")
    sep()
    print("""
  Confidence bands on |conf_diff|  (conf_diff = prob − threshold):
    borderline  |conf_diff| < 0.15  — model was uncertain, close call
    moderate    0.15 ≤ |conf_diff| < 0.35  — model leaned the wrong way
    misfire     |conf_diff| ≥ 0.35  — model was very confident and wrong
    """)
    for _, r in s2.iterrows():
        print(f"  {r.appliance.upper():<20} {r.error_type}  "
              f"n={r.n_errors:>4}  mean_conf_diff={r.mean_conf_diff:+.3f}")
        print(f"    borderline : {r.borderline_n:>4}  ({r.borderline_pct:5.1f}%)")
        print(f"    moderate   : {r.moderate_n:>4}  ({r.moderate_pct:5.1f}%)")
        print(f"    misfire    : {r.misfire_n:>4}  ({r.misfire_pct:5.1f}%)")
        print()


# ─── HELPERS ─────────────────────────────────────────────────────────────

def sep():
    print("\n" + "=" * 70)


# ─── MAIN ────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python eda_heldout.py <failure_analysis.csv>")
        sys.exit(1)

    path = Path(sys.argv[1])
    stem = path.stem
    out_dir = path.parent

    print(f"\nFile : {path}")
    df = load(path)
    df = add_complexity_flag(df)
    print(f"Rows : {len(df)}  |  Cols : {len(df.columns)}")

    # build tables
    s0 = build_s0(df)
    s1 = build_s1(df)
    s2 = build_s2(df)

    # print
    section_overview(df, s0)
    section_performance_by_flag(df, s1)
    section_confidence(df, s2)

    # save CSVs
    p0 = out_dir / f"{stem}_s0_complexity.csv"
    p1 = out_dir / f"{stem}_s1_performance.csv"
    p2 = out_dir / f"{stem}_s2_confidence.csv"

    s0.to_csv(p0, index=False)
    s1.to_csv(p1, index=False)
    s2.to_csv(p2, index=False)

    sep()
    print("CSVs saved:")
    print(f"  {p0}")
    print(f"  {p1}")
    print(f"  {p2}")
    print()


if __name__ == "__main__":
    main()