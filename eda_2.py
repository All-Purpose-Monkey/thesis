import numpy as np
import pandas as pd
import os
from preprocess import preprocess_nilm_states
import matplotlib.pyplot as plt

def eda_power_buckets(df, appliance_cols):

    appliance_cols = [c for c in appliance_cols if c in df.columns]

    # clean numeric
    for c in appliance_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # define fixed NILM energy basins
    bins = [0, 1, 5, 50, 200, 1000, 2500, np.inf]
    labels = [
        "0",
        "1-5",
        "6-50",
        "51-200",
        "201-1000",
        "1000-2500",
        "2500+"
    ]

    results = {}

    for col in appliance_cols:

        data = df[col].dropna().values

        if len(data) == 0:
            continue

        # binning
        binned = pd.cut(data, bins=bins, labels=labels, include_lowest=True)

        counts = binned.value_counts().sort_index()
        percent = (counts / counts.sum()) * 100

        results[col] = pd.DataFrame({
            "count": counts,
            "percent": percent
        })

        # print
        print(f"\n================ {col} ================")

        for label in labels:
            print(f"{label:10s} | count: {counts[label]:10d} | {percent[label]:6.2f}%")

    return results

def eda_nilm_summary(df, appliance_cols):

    df = df.copy()

    appliance_cols = [c for c in appliance_cols if c in df.columns]

    # ensure numeric

    for c in appliance_cols:

        df[c] = pd.to_numeric(df[c], errors="coerce")

    # fill NaNs as 0 for activity counting ONLY (important for NILM stats)

    active = df[appliance_cols].fillna(0)

    # ----------------------------

    # 1. TOTAL ROWS

    # ----------------------------

    total_rows = len(df)

    table_1 = pd.DataFrame({

        "metric": ["total_rows"],

        "value": [total_rows]

    })

    # ----------------------------

    # 2. ROW ACTIVITY LEVELS

    # ----------------------------

    active_count = active.sum(axis=1)

    no_active = (active_count == 0).sum()

    one_active = (active_count == 1).sum()

    multi_active = (active_count > 1).sum()

    table_2 = pd.DataFrame({

        "state": ["no appliance active", "1 appliance active", "multiple appliances active"],

        "count": [no_active, one_active, multi_active],

        "percent": [

            no_active / total_rows * 100,

            one_active / total_rows * 100,

            multi_active / total_rows * 100

        ]

    })

    # ----------------------------

    # 3. PER-APPLIANCE ACTIVITY

    # ----------------------------

    inactive_counts = (active == 0).sum()

    active_counts = (active == 1).sum()

    table_3 = pd.DataFrame({

        "appliance": appliance_cols,

        "inactive_count": inactive_counts.values,

        "active_count": active_counts.values,

        "inactive_%": inactive_counts.values / total_rows * 100,

        "active_%": active_counts.values / total_rows * 100

    })
    return table_1, table_2, table_3

def add_nilm_flags_and_save(df, appliance_cols, save_path):

    df = df.copy()
    appliance_cols = [c for c in appliance_cols if c in df.columns]

    # ----------------------------
    # ensure datetime index
    # ----------------------------

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df.set_index("datetime")

    # ----------------------------
    # 1. GLOBAL ACTIVE FLAG
    # ----------------------------

    active_matrix = df[appliance_cols].fillna(0)
    df["is_active"] = (active_matrix.sum(axis=1) > 0).astype(int)

    # ----------------------------
    # 2. DAY TIME BUCKETS
    # ----------------------------

    hours = df.index.hour

    df["day_time"] = np.select(
          [
            (hours >= 0) & (hours < 8),
            (hours >= 8) & (hours < 12),
            (hours >= 12) & (hours < 17),
            (hours >= 17) & (hours < 24),
        ],
        [0, 1, 2, 3],
        default=-1
    )

    # ----------------------------
    # 3. APPLIANCE ACTIVE SCORE - doubles up as a example difficulty score for NILM (more active appliances = harder to disaggregate)
    # ----------------------------

    df["active_count"] = active_matrix.sum(axis=1)

    save_path = os.path.expanduser(save_path)
    df.to_csv(save_path)
    print(f"Saved to: {save_path}")

    return df

def eda_activity_by_daytime(csv_path):

    csv_path = os.path.expanduser(csv_path)

    df = pd.read_csv(csv_path)

    # detect appliance columns (binary ones)

    appliance_cols = [c for c in df.columns if c.startswith("ch")]

    results = []

    for col in appliance_cols:

        # group by day_time

        grouped = df.groupby("day_time")[col]

        for dt, values in grouped:

            total = len(values)

            active = (values == 1).sum()

            inactive = (values == 0).sum()

            results.append({

                "appliance": col,

                "day_time": dt,

                "active_count": active,

                "inactive_count": inactive,

                "active_%": (active / total) * 100 if total > 0 else 0,

                "inactive_%": (inactive / total) * 100 if total > 0 else 0

            })

    result_df = pd.DataFrame(results)

    # sort nicely

    result_df = result_df.sort_values(["appliance", "day_time"])

    print("\n--- ACTIVITY BY DAY TIME PER APPLIANCE ---")

    print(result_df)

    return result_df


def run_label_eda(
    house,
    labels_path,
    output_dir="~/thesis/graphs"
):
    """
    EDA for UK-DALE label files.

    Produces:
    1. Null vs non-null counts per column
    2. Null counts by year/month per column
    3. Histograms (10 bins) per numeric column

    Histograms are saved to:
    ~/thesis/graphs/house_X/
    """

    # Expand paths
    labels_path = os.path.expanduser(labels_path)

    output_dir = os.path.expanduser(
        f"{output_dir}/house_{house}"
    )

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(labels_path)

    print(f"\nLoaded shape: {df.shape}")

    # --------------------------------------------------
    # TIMESTAMP HANDLING
    # --------------------------------------------------

    possible_ts_cols = [
        "timestamp",
        "time",
        "datetime",
        "DateTime",
        "utc_timestamp"
    ]

    ts_col = None

    for col in possible_ts_cols:
        if col in df.columns:
            ts_col = col
            break

    if ts_col is not None:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

        df["year"] = df[ts_col].dt.year
        df["month"] = df[ts_col].dt.month

        print(f"Using timestamp column: {ts_col}")

    else:
        print("No timestamp column detected.")

    # --------------------------------------------------
    # 1. NULL VS NON-NULL
    # --------------------------------------------------

    print("\n==============================")
    print("NULL VS NON-NULL BY COLUMN")
    print("==============================")

    null_summary = pd.DataFrame({
        "nulls": df.isnull().sum(),
        "non_nulls": df.notnull().sum()
    })

    print(null_summary)

    null_summary.to_csv(
        os.path.join(output_dir, "null_vs_nonnull.csv")
    )

    # --------------------------------------------------
    # 2. NULLS BY YEAR/MONTH
    # --------------------------------------------------

    if ts_col is not None:

        print("\n==============================")
        print("NULLS BY YEAR/MONTH")
        print("==============================")

        monthly_nulls = []

        for col in df.columns:

            if col in ["year", "month"]:
                continue

            tmp = (
                df.groupby(["year", "month"])[col]
                .apply(lambda x: x.isnull().sum())
                .reset_index(name="null_count")
            )

            tmp["column"] = col

            monthly_nulls.append(tmp)

        monthly_nulls_df = pd.concat(
            monthly_nulls,
            ignore_index=True
        )

        print(monthly_nulls_df.head())

        monthly_nulls_df.to_csv(
            os.path.join(output_dir, "monthly_nulls.csv"),
            index=False
        )

    # --------------------------------------------------
    # 3. HISTOGRAMS
    # --------------------------------------------------

    print("\n==============================")
    print("GENERATING HISTOGRAMS")
    print("==============================")

    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns

    for col in numeric_cols:

        if col in ["year", "month"]:
            continue

        values = df[col].dropna()

        if len(values) == 0:
            print(f"Skipping empty column: {col}")
            continue

        plt.figure(figsize=(8, 5))

        plt.hist(values, bins=10)

        plt.title(f"House {house} - {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")

        save_path = os.path.join(
            output_dir,
            f"{col}_hist.png"
        )

        plt.savefig(save_path, bbox_inches="tight")

        plt.close()

        print(f"Saved: {save_path}")

    print("\nEDA COMPLETE.")

#if __name__ == "__main__":

    #csv_path = os.path.expanduser("~/thesis/house_1/house1_stitched.csv")
    #df = pd.read_csv(csv_path)

    #appliance_cols = [c for c in df.columns if c.startswith("ch")]

    #results = eda_power_buckets(df, appliance_cols)

    #df_binary = preprocess_nilm_states(df, appliance_cols)

    #t1, t2, t3 = eda_nilm_summary(df_binary, appliance_cols)

    #print("\n--- TOTAL ROWS ---")
    #print(t1)

    #print("\n--- ROW ACTIVITY LEVELS ---")
    #print(t2)

    #print("\n--- PER APPLIANCE ACTIVITY ---")
    #print(t3)

    #df_out = add_nilm_flags_and_save(df_binary, appliance_cols, "~/thesis/house_1/house1_binarized.csv")

    #csv_path = os.path.expanduser("~/thesis/house_1/house1_binarized.csv")
    #df = pd.read_csv(csv_path)
    #eda_activity_by_daytime("~/thesis/house_1/house1_binarized.csv")
run_label_eda(
    house=1,
    labels_path="~/thesis/data/house_1/house1_stitched.csv"
)
run_label_eda(
    house=2,
    labels_path="~/thesis/data/house_2/house2_stitched.csv"
)
run_label_eda(
    house=5,
    labels_path="~/thesis/data/house_5/house5_stitched.csv"
)