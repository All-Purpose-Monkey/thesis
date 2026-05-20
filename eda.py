import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run_label_eda(
    house,
    labels_path,
    output_dir="~/thesis/graphs",
    hist_bucket_size=10,
    on_threshold=5,
):
    """
    EDA for UK-DALE label files.

    Produces:
    1. Null vs non-null counts per column             → null_vs_nonnull.csv
    2. Null counts by year/month per column           → monthly_nulls_pivot.csv
    3. Histograms (adaptive bins, log y-scale)        → {col}_hist.png
    4. Histogram bin stats as CSV                     → {col}_hist_stats.csv
       (bin_start, bin_end, bin_mid, count, cumulative_pct)
       Use this to determine on/off thresholds without reading the PNG.
    5. Monthly on-state counts (power > on_threshold) → monthly_on_counts_pivot.csv
       Same pivot shape as nulls: rows = year-month, cols = appliances.

    Args:
        house:            int, house number (used for output subdirectory and plot titles)
        labels_path:      str, path to the stitched CSV (~ expanded automatically)
        output_dir:       str, base output directory (default ~/thesis/graphs)
        hist_bucket_size: int/float, fixed bin width for histograms (default 10W)
        on_threshold:     float, watts above which a reading is counted as "on" (default 5)

    Saves all outputs to:
        {output_dir}/house_{house}/
    """

    # --------------------------------------------------
    # PATHS
    # --------------------------------------------------

    labels_path = os.path.expanduser(labels_path)

    output_dir = os.path.expanduser(
        os.path.join(output_dir, f"house_{house}")
    )

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------
    # LOAD CSV
    # --------------------------------------------------

    df = pd.read_csv(labels_path)

    print(f"\n[House {house}] Loaded shape: {df.shape}")

    # --------------------------------------------------
    # TIMESTAMP DETECTION
    # --------------------------------------------------

    possible_ts_cols = [
        "timestamp", "time", "datetime", "DateTime", "utc_timestamp"
    ]

    ts_col = None

    for col in possible_ts_cols:
        if col in df.columns:
            ts_col = col
            break

    if ts_col is not None:

        print(f"[House {house}] Using timestamp column: {ts_col}")

        df[ts_col] = pd.to_datetime(
            df[ts_col],
            unit="s",
            utc=True,
            errors="coerce",
        )

        df["year"]  = df[ts_col].dt.year
        df["month"] = df[ts_col].dt.month

    else:
        print(f"[House {house}] No timestamp column found — skipping time-based analyses.")

    # Columns that are appliance readings (not meta columns)
    meta_cols = {ts_col, "year", "month"} if ts_col else {"year", "month"}
    analysis_cols = [c for c in df.columns if c not in meta_cols]

    # --------------------------------------------------
    # 1. NULL VS NON-NULL
    # --------------------------------------------------

    print(f"\n[House {house}] NULL VS NON-NULL BY COLUMN")

    null_summary = pd.DataFrame({
        "nulls":     df[analysis_cols].isnull().sum(),
        "non_nulls": df[analysis_cols].notnull().sum(),
        "null_pct":  (df[analysis_cols].isnull().sum() / len(df) * 100).round(2),
    })

    print(null_summary.to_string())

    null_summary.to_csv(
        os.path.join(output_dir, "null_vs_nonnull.csv")
    )

    # --------------------------------------------------
    # 2. NULLS BY YEAR/MONTH (PIVOTED)
    # --------------------------------------------------

    if ts_col is not None:

        print(f"\n[House {house}] NULLS BY YEAR/MONTH (PIVOTED)")

        monthly_nulls = (
            df.groupby(["year", "month"])[analysis_cols]
            .apply(lambda x: x.isnull().sum())
            .reset_index()
        )

        pivot_nulls = monthly_nulls.set_index(["year", "month"])

        print(pivot_nulls.head().to_string())

        pivot_nulls.to_csv(
            os.path.join(output_dir, "monthly_nulls_pivot.csv")
        )

    # --------------------------------------------------
    # 3 + 4. HISTOGRAMS + HISTOGRAM CSV STATS
    # --------------------------------------------------

    print(f"\n[House {house}] GENERATING HISTOGRAMS + STATS CSVs")

    numeric_cols = df[analysis_cols].select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:

        values = df[col].dropna()

        if len(values) == 0:
            print(f"  Skipping empty column: {col}")
            continue

        vmin = values.min()
        vmax = values.max()

        if vmin == vmax:
            vmax = vmin + 1

        bins = np.arange(vmin, vmax + hist_bucket_size, hist_bucket_size)

        if len(bins) < 2:
            bins = 10

        # --- Compute histogram counts (shared between PNG and CSV) ---
        counts, bin_edges = np.histogram(values, bins=bins)

        # ---- 3. PNG histogram ----
        plt.figure(figsize=(8, 5))
        plt.hist(values, bins=bin_edges)
        plt.yscale("log")
        plt.title(f"House {house} - {col}")
        plt.xlabel(f"{col} (W)")
        plt.ylabel("Frequency (log scale)")
        plt.xlim(vmin, vmax)

        png_path = os.path.join(output_dir, f"{col}_hist.png")
        plt.savefig(png_path, bbox_inches="tight")
        plt.close()

        print(f"  Saved PNG: {png_path}")

        # ---- 4. CSV of histogram bin stats ----
        # Gives you the exact numbers behind each bar so you can
        # pick on/off thresholds programmatically rather than
        # eyeballing the PNG.
        #
        # Columns:
        #   bin_start      — left edge of bin (inclusive)
        #   bin_end        — right edge of bin (exclusive, last bin inclusive)
        #   bin_mid        — midpoint, convenient for plotting / lookups
        #   count          — number of samples in this bin
        #   pct            — percentage of non-null samples in this bin
        #   cumulative_pct — running total %; useful for finding where
        #                    95%+ of the "off" mass ends

        total_samples = counts.sum()
        pct = (counts / total_samples * 100).round(4) if total_samples > 0 else counts * 0.0
        cumulative_pct = np.cumsum(pct).round(4)

        hist_stats = pd.DataFrame({
            "bin_start":      bin_edges[:-1].round(4),
            "bin_end":        bin_edges[1:].round(4),
            "bin_mid":        ((bin_edges[:-1] + bin_edges[1:]) / 2).round(4),
            "count":          counts,
            "pct":            pct,
            "cumulative_pct": cumulative_pct,
        })

        # Flag bins that are above the on_threshold so it's obvious
        # in the CSV which region is "on" vs "off + ambient"
        hist_stats["region"] = np.where(
            hist_stats["bin_mid"] > on_threshold, "on", "off_or_ambient"
        )

        csv_path = os.path.join(output_dir, f"{col}_hist_stats.csv")
        hist_stats.to_csv(csv_path, index=False)

        print(f"  Saved CSV: {csv_path}")

    # --------------------------------------------------
    # 5. MONTHLY ON-STATE COUNTS (PIVOTED)
    # on = power reading strictly above on_threshold
    # --------------------------------------------------

    if ts_col is not None:

        print(f"\n[House {house}] MONTHLY ON-STATE COUNTS (threshold > {on_threshold}W)")

        # For each appliance column, create a boolean "is on" series
        on_df = df[["year", "month"]].copy()

        for col in numeric_cols:
            # NaN rows contribute 0 to the on count (they are not "on")
            on_df[col] = (df[col].fillna(0) > on_threshold).astype(int)

        monthly_on = (
            on_df.groupby(["year", "month"])[numeric_cols]
            .sum()
        )

        # monthly_on rows = (year, month), cols = appliances
        # Values = number of 6-second windows where appliance was "on"
        # Divide by 10 to get approximate minutes, or leave as counts

        print(monthly_on.head().to_string())

        monthly_on.to_csv(
            os.path.join(output_dir, "monthly_on_counts_pivot.csv")
        )

        # Also save a "minutes on per month" version (assuming 6s windows)
        monthly_on_mins = (monthly_on * 6 / 60).round(1)
        monthly_on_mins.to_csv(
            os.path.join(output_dir, "monthly_on_minutes_pivot.csv")
        )

        print(f"  Saved: monthly_on_counts_pivot.csv")
        print(f"  Saved: monthly_on_minutes_pivot.csv")

    print(f"\n[House {house}] EDA complete. Outputs in: {output_dir}\n")


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    run_label_eda(
        house=1,
        labels_path="~/thesis/data/house_1/house1_stitched.csv",
        hist_bucket_size=10,
        on_threshold=5,
    )

    run_label_eda(
        house=2,
        labels_path="~/thesis/data/house_2/house2_stitched.csv",
        hist_bucket_size=10,
        on_threshold=5,
    )

    run_label_eda(
        house=5,
        labels_path="~/thesis/data/house_5/house5_stitched.csv",
        hist_bucket_size=10,
        on_threshold=5,
    )