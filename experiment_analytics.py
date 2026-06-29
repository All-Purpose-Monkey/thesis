import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


def plot_channel_ablation(path_current, path_voltage, out_dir="~/thesis/results/finale/"):
    path_current = os.path.expanduser(path_current)
    path_voltage = os.path.expanduser(path_voltage)
    out_dir      = os.path.expanduser(out_dir)

    df_c = pd.read_csv(path_current)
    df_v = pd.read_csv(path_voltage)

    for df in [df_c, df_v]:
        df["appliance"] = df["appliance"].str.replace("TOTAL (macro avg)", "macro avg", regex=False)

    order = [a for a in df_c["appliance"] if a != "macro avg"] + ["macro avg"]
    df_c  = df_c.set_index("appliance").loc[order]
    df_v  = df_v.set_index("appliance").loc[order]

    x     = np.arange(len(order))
    width = 0.18

    COLOR_CURRENT = "#2c7bb6"
    COLOR_VOLTAGE = "#d7191c"

    fig, ax = plt.subplots(figsize=(14, 5))

    configs = [
        ("f1",      COLOR_CURRENT, "",     "F1 — current",  -1.5),
        ("f1",      COLOR_VOLTAGE, "////", "F1 — voltage",  -0.5),
        ("roc_auc", COLOR_CURRENT, "....", "AUC — current",  0.5),
        ("roc_auc", COLOR_VOLTAGE, "xxxx", "AUC — voltage",  1.5),
    ]

    for metric, color, hatch, label, offset in configs:
        src  = df_c if "current" in label else df_v
        vals = src[metric].values.astype(float)
        # skip AUC bars where NaN (dishwasher, macro avg) rather than zero-filling
        plot_vals = np.where(np.isnan(vals), 0.0, vals)
        bars = ax.bar(x + offset * width, plot_vals, width,
                      color=color, hatch=hatch, alpha=0.8,
                      edgecolor="white", label=label)

    ax.axvline(x=len(order) - 1.5, color="grey", linestyle=":", linewidth=1.2, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylabel("Score")
    ax.set_title("Channel Ablation — F1 & AUC per Appliance")
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    out = os.path.join(out_dir, "channel_ablation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_baseline_test_results(path, out_dir="~/thesis/results/finale/"):
    path    = os.path.expanduser(path)
    out_dir = os.path.expanduser(out_dir)

    df = pd.read_csv(path)
    df["appliance"] = df["appliance"].str.replace("TOTAL (macro avg)", "macro", regex=False)

    order   = [a for a in df["appliance"] if a != "macro"] + ["macro"]
    df      = df.set_index("appliance").loc[order]

    metrics = ["f1", "precision", "recall", "roc_auc"]
    colors  = ["#2c7bb6", "#fdae61", "#1a9641", "#d7191c"]
    x       = np.arange(len(order))
    width   = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    fig, ax = plt.subplots(figsize=(14, 5))

    for offset, metric, color in zip(offsets, metrics, colors):
        vals = df[metric].values.astype(float)
        bars = ax.bar(x + offset * width, vals, width,
                      color=color, alpha=0.85, label=metric, edgecolor="white")
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=6.5)

    ax.axvline(x=len(order) - 1.5, color="grey", linestyle=":", linewidth=1.2, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylabel("Score")
    ax.set_title("Baseline CNN — Test Results per Appliance")
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    out = os.path.join(out_dir, "baseline_test_results.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_training_history(path, out_dir="~/thesis/results/finale/"):
    path    = os.path.expanduser(path)
    out_dir = os.path.expanduser(out_dir)

    df         = pd.read_csv(path)
    appliances = ["kettle", "toaster", "microwave", "dishwasher", "fridge", "washing_machine"]
    auc_cols   = [f"{a}_roc_auc_score" for a in appliances]
    f1_cols    = [f"{a}_f1" for a in appliances]
    epochs     = df["epoch"].values
    colors     = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training Trajectories", fontsize=14, y=1.01)

    handles, labels = [], []

    for ax, cols, ylabel, title in [
        (axes[0], auc_cols, "ROC-AUC", "ROC-AUC History"),
        (axes[1], f1_cols,  "F1",      "F1 History"),
    ]:
        for col, appliance, color in zip(cols, appliances, colors):
            line, = ax.plot(epochs, df[col].values,
                            color=color, linewidth=1.4, alpha=0.85)
            if ax is axes[0]:
                handles.append(line)
                labels.append(appliance.replace("_", " "))

        # macro line
        macro_vals = df[cols].mean(axis=1) if ax is axes[0] else df["mean_f1"]
        mline, = ax.plot(epochs, macro_vals, color="black", linewidth=2, linestyle="--")
        if ax is axes[0]:
            handles.append(mline)
            labels.append("macro avg")

        # best F1 marker on F1 subplot
        if ax is axes[1]:
            best_f1   = df["mean_f1"].max()
            best_epoch = df.loc[df["mean_f1"].idxmax(), "epoch"]
            ax.axhline(best_f1, color="crimson", linewidth=1.2, linestyle=":")
            ax.axvline(best_epoch, color="crimson", linewidth=1.2, linestyle=":",
                       label=f"best (ep {best_epoch}, F1={best_f1:.3f})")
            ax.legend(fontsize=8, loc="lower right")

        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.grid(linestyle="--", alpha=0.4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    # loss subplot
    axes[2].plot(epochs, df["train_loss"].values, color="black", linewidth=1.8, label="train_loss")
    axes[2].plot(epochs, df["val_loss"].values, color="crimson", linewidth=1.8,
                 linestyle="--", label="val_loss")
    axes[2].legend(fontsize=8)
    axes[2].grid(linestyle="--", alpha=0.4)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Loss")

    # shared legend below the auc + f1 subplots
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               fontsize=9, bbox_to_anchor=(0.38, -0.08), frameon=True)

    fig.tight_layout()
    out = os.path.join(out_dir, "baseline_training_history.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_padding_ablation(path, out_dir="~/thesis/results/finale/"):
    path    = os.path.expanduser(path)
    out_dir = os.path.expanduser(out_dir)

    df = pd.read_csv(path)

    # clean config labels
    df["label"] = df.apply(
        lambda r: f"pad={int(r['padding'])}, sym={r['sym_pad']}", axis=1
    )

    x      = np.arange(len(df))
    width  = 0.3
    colors = ["#2c7bb6", "#d7191c"]

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (metric, color, label) in enumerate([
        ("macro_f1",  colors[0], "F1"),
        ("macro_auc", colors[1], "AUC"),
    ]):
        bars = ax.bar(x + (i - 0.5) * width, df[metric].values, width,
                      color=color, alpha=0.85, label=label, edgecolor="white")
        for bar, val in zip(bars, df[metric].values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0.88, 1.0)
    ax.set_yticks(np.arange(0.88, 1.01, 0.02))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylabel("Score")
    ax.set_title("Padding Ablation — F1 & AUC per Config")
    ax.legend(fontsize=9)

    fig.tight_layout()
    out = os.path.join(out_dir, "padding_ablation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def _pool_group(pool_str):
    """Assign a shape group label to a pool size string like '(3, 2)'."""
    vals  = [int(v) for v in pool_str.strip("()").split(",")]
    pf, pt = vals[0], vals[1]
    if pf == pt:           return "square"
    elif pt == 1:          return "freq-only"
    elif pf == 1:          return "time-only"
    elif pf > pt:          return "freq-dominant"
    else:                  return "time-dominant"


def plot_pool_scatter(path, out_dir="~/thesis/results/finale/"):
    path    = os.path.expanduser(path)
    out_dir = os.path.expanduser(out_dir)

    df = pd.read_csv(path)
    df["group"] = df["pool_size"].apply(_pool_group)

    groups      = ["freq-only", "freq-dominant", "square", "time-dominant", "time-only"]
    group_colors = {g: c for g, c in zip(groups, plt.cm.tab10.colors)}

    fig, ax = plt.subplots(figsize=(10, 6))

    for group in groups:
        sub = df[df["group"] == group].sort_values("embed_dim")
        color = group_colors[group]
        ax.scatter(sub["embed_dim"], sub["macro_f1"],
                   color=color, label=group,
                   s=90, edgecolors="white", linewidths=0.6, zorder=3)
        for _, row in sub.iterrows():
            ax.annotate(row["pool_size"], (row["embed_dim"], row["macro_f1"]),
                        textcoords="offset points", xytext=(5, 3), fontsize=7, alpha=0.8)

    ax.set_ylim(0.87, 0.96)
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Macro F1")
    ax.set_title("Pool Ablation — F1 vs Embedding Size")
    ax.legend(title="Pool Shape", fontsize=9)

    fig.tight_layout()
    out = os.path.join(out_dir, "pool_ablation_scatter.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_pool_best_configs(summary_path, results_map, out_dir="~/thesis/results/finale/"):
    """
    Per-appliance F1 bars for the best pool config in each shape group.

    results_map: dict of {pool_size_string: csv_path}
      e.g. {"(12, 1)": "~/thesis/results/finale/pool_12_1_test_results.csv", ...}
      Each CSV: appliance, f1, precision, recall, roc_auc (same format as baseline test results)
    """
    summary_path = os.path.expanduser(summary_path)
    out_dir      = os.path.expanduser(out_dir)

    df = pd.read_csv(summary_path)
    df["group"] = df["pool_size"].apply(_pool_group)

    groups       = ["freq-only", "freq-dominant", "square", "time-dominant", "time-only"]
    group_colors = {g: c for g, c in zip(groups, plt.cm.tab10.colors)}
    best         = df.loc[df.groupby("group")["macro_f1"].idxmax()].set_index("group")

    appliances = ["kettle", "toaster", "microwave", "dishwasher", "fridge", "washing_machine"]
    x       = np.arange(len(appliances))
    width   = 0.15
    offsets = np.linspace(-(len(groups) - 1) / 2, (len(groups) - 1) / 2, len(groups)) * width

    fig, ax = plt.subplots(figsize=(14, 5))
    plotted = 0

    for offset, group in zip(offsets, groups):
        if group not in best.index:
            continue
        pool_size = best.loc[group, "pool_size"].strip()
        csv_path  = os.path.expanduser(results_map.get(pool_size, ""))
        if not csv_path or not os.path.exists(csv_path):
            print(f"[SKIP] {group} best={pool_size} — no CSV at {csv_path}")
            continue

        res = pd.read_csv(csv_path)
        res = res[~res["appliance"].str.contains("macro|TOTAL", case=False)].set_index("appliance")
        vals  = [res.loc[a, "f1"] if a in res.index else 0.0 for a in appliances]
        # clean label: "(12, 1)" → "12×1"
        clean = pool_size.strip("()").replace(", ", "×")
        ax.bar(x + offset, vals, width,
               color=group_colors[group], alpha=0.85,
               label=f"{group} [{clean}]", edgecolor="white")
        plotted += 1

    if plotted == 0:
        print("[plot_pool_best_configs] No CSVs found — update results_map paths.")

    ax.set_xticks(x)
    ax.set_xticklabels(appliances, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylabel("F1")
    ax.set_title("Pool Ablation — Best Config per Group, F1 by Appliance")
    ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    out = os.path.join(out_dir, "pool_best_configs.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


plot_channel_ablation(
    "~/thesis/results/finale/test_results_current.csv",
    "~/thesis/results/finale/test_results_voltage.csv",
)

plot_baseline_test_results("~/thesis/results/finale/cnn_baseline_test_results_final.csv")

plot_training_history("~/thesis/results/finale/cnn_baseline_nothresholds_history.csv")

plot_padding_ablation("~/thesis/results/finale/padding_ablation_summary.csv")

plot_pool_scatter("~/thesis/results/finale/pool_ablation_summary.csv")

# fill in the per-appliance CSVs for each best config before running this
POOL_RESULTS_MAP = {
    "(12, 1)": "~/thesis/results/finale/test_results_12_1.csv",   # freq-only best
    "(3, 2)":  "~/thesis/results/finale/test_results_3_2.csv",    # freq-dominant best
    "(6, 6)":  "~/thesis/results/finale/test_results_6_6.csv",    # square best
    "(2, 3)":  "~/thesis/results/finale/test_results_2_3.csv",    # time-dominant best
    "(1, 12)": "~/thesis/results/finale/test_results_1_12.csv",   # time-only best
}
plot_pool_best_configs("~/thesis/results/finale/pool_ablation_summary.csv", POOL_RESULTS_MAP)


def _compression_pool_family(pool_str):
    """Classify pool size into one of three adpool families."""
    vals = [int(v) for v in pool_str.strip("()").split(",")]
    pf, pt = vals[0], vals[1]
    if pf == 1 and pt == 1:
        return "(1,1)"
    elif pt == 1:
        return "freq-only"
    else:
        return "freq-dominant"


def plot_compression_scatter(path1, path2, out_dir="~/thesis/results/finale/"):
    path1   = os.path.expanduser(path1)
    path2   = os.path.expanduser(path2)
    out_dir = os.path.expanduser(out_dir)

    df = pd.concat([pd.read_csv(path1), pd.read_csv(path2)], ignore_index=True)
    df["family"] = df["pool_size"].apply(_compression_pool_family)

    # color = adpool family
    family_colors = {"freq-only": "#2c7bb6", "freq-dominant": "#1a9641", "(1,1)": "#d7191c"}
    # shape = kernel size
    kernel_markers = {3: "o", 5: "s", 7: "^"}
    # fill = stride: stride=2 filled, stride=3 hollow
    stride_fill = {2: True, 3: False}

    fig, ax = plt.subplots(figsize=(12, 6))

    # build legend handles manually to avoid explosion of entries
    from matplotlib.lines import Line2D
    legend_handles = []
    for family, color in family_colors.items():
        legend_handles.append(Line2D([0], [0], marker="o", color="w",
                                     markerfacecolor=color, markersize=9, label=f"pool: {family}"))
    for kernel, marker in kernel_markers.items():
        legend_handles.append(Line2D([0], [0], marker=marker, color="grey",
                                     markersize=9, label=f"kernel {kernel}×{kernel}",
                                     markerfacecolor="grey"))
    legend_handles.append(Line2D([0], [0], marker="o", color="w",
                                 markerfacecolor="grey", markersize=9, label="stride 2 (filled)"))
    legend_handles.append(Line2D([0], [0], marker="o", color="grey",
                                 markerfacecolor="none", markersize=9,
                                 markeredgewidth=1.5, label="stride 3 (hollow)"))

    for family, color in family_colors.items():
        for kernel, marker in kernel_markers.items():
            for stride, filled in stride_fill.items():
                sub = df[(df["family"] == family) & (df["kernel"] == kernel) & (df["stride"] == stride)]
                if sub.empty:
                    continue
                ax.scatter(sub["n_params"], sub["macro_f1"],
                           color=color if filled else "none",
                           edgecolors=color,
                           marker=marker, s=90,
                           linewidths=1.5, zorder=3)
                for _, row in sub.iterrows():
                    clean = row["pool_size"].strip("()").replace(", ", "×")
                    ax.annotate(clean, (row["n_params"], row["macro_f1"]),
                                textcoords="offset points", xytext=(5, 3),
                                fontsize=6.5, alpha=0.8)

    ax.grid(linestyle="--", alpha=0.4)
    ax.set_xlabel("Trainable Parameters")
    ax.set_ylabel("Macro F1")
    ax.set_title("Compression Ablation — F1 vs Parameters")
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

    fig.tight_layout()
    out = os.path.join(out_dir, "compression_scatter.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_compression_best5(path1, path2, results_map, out_dir="~/thesis/results/finale/"):
    """
    Per-appliance F1 bars for the top 5 configs by macro F1.

    results_map: dict of {config_name: csv_path}
      config_name matches the 'config' column, e.g. "pool_9_1_stride_3_kernel_5"
      CSV format: appliance, f1, precision, recall, roc_auc
    """
    path1   = os.path.expanduser(path1)
    path2   = os.path.expanduser(path2)
    out_dir = os.path.expanduser(out_dir)

    df   = pd.concat([pd.read_csv(path1), pd.read_csv(path2)], ignore_index=True)
    top5 = df.nlargest(5, "macro_f1").reset_index(drop=True)

    appliances = ["kettle", "toaster", "microwave", "dishwasher", "fridge", "washing_machine"]
    x       = np.arange(len(appliances))
    width   = 0.15
    offsets = np.linspace(-2, 2, 5) * width
    colors  = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(14, 5))

    for offset, (_, row), color in zip(offsets, top5.iterrows(), colors):
        cfg      = row["config"]
        csv_path = os.path.expanduser(results_map.get(cfg, ""))
        if not csv_path or not os.path.exists(csv_path):
            print(f"[SKIP] {cfg} — no CSV at {csv_path}")
            continue

        res  = pd.read_csv(csv_path)
        res  = res[~res["appliance"].str.contains("macro|TOTAL", case=False)].set_index("appliance")
        vals = [res.loc[a, "f1"] if a in res.index else 0.0 for a in appliances]

        clean = f"{row['pool_size'].strip('()').replace(', ', '×')} s={row['stride']} k={row['kernel']}"
        ax.bar(x + offset, vals, width, color=color, alpha=0.85,
               label=clean, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(appliances, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylabel("F1")
    ax.set_title("Compression Ablation — Top 5 Configs, F1 by Appliance")
    ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    out = os.path.join(out_dir, "compression_best5.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


plot_compression_scatter(
    "~/thesis/results/finale/compression_ablation_summary.csv",
    "~/thesis/results/finale/compression_ablation_summary_2.csv",
)

COMPRESSION_RESULTS_MAP = {
    "pool_9_1_stride_3_kernel_5": "~/thesis/results/finale/test_results_pool_9_1_stride_3_kernel_5.csv",
    "pool_9_1_stride_3_kernel_7": "~/thesis/results/finale/test_results_pool_9_1_stride_3_kernel_7.csv",
    "pool_6_1_stride_2_kernel_7": "~/thesis/results/finale/test_results_pool_6_1_stride_2_kernel_7.csv",
    "pool_6_3_stride_3_kernel_7": "~/thesis/results/finale/test_results_pool_6_3_stride_3_kernel_7.csv",
    "pool_6_1_stride_3_kernel_7": "~/thesis/results/finale/test_results_pool_6_1_stride_3_kernel_7.csv",
}

plot_compression_best5(
    "~/thesis/results/finale/compression_ablation_summary.csv",
    "~/thesis/results/finale/compression_ablation_summary_2.csv",
    COMPRESSION_RESULTS_MAP,
)


def plot_heldout_summary(perf_path, raw_path, out_dir="~/thesis/results/finale/"):
    perf_path = os.path.expanduser(perf_path)
    raw_path  = os.path.expanduser(raw_path)
    out_dir   = os.path.expanduser(out_dir)

    perf = pd.read_csv(perf_path)
    raw  = pd.read_csv(raw_path)

    appliances = ["kettle", "toaster", "microwave", "dishwasher", "fridge", "washing_machine"]
    agg = perf[perf["complexity"] != "TOTAL"].groupby("appliance")[["TP", "FP", "FN"]].sum()

    rows = []
    for a in appliances:
        tp, fp, fn = agg.loc[a, "TP"], agg.loc[a, "FP"], agg.loc[a, "FN"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        y_true = raw[f"{a}_true"].values
        y_prob = raw[f"{a}_prob"].values
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
        rows.append({"appliance": a, "f1": f1, "precision": prec, "recall": rec, "roc_auc": auc})

    df = pd.DataFrame(rows)
    macro = {"appliance": "macro avg", "f1": df["f1"].mean(), "precision": df["precision"].mean(),
             "recall": df["recall"].mean(), "roc_auc": df["roc_auc"].mean()}
    df = pd.concat([df, pd.DataFrame([macro])], ignore_index=True)

    order   = appliances + ["macro avg"]
    metrics = ["f1", "precision", "recall", "roc_auc"]
    colors  = ["#2c7bb6", "#fdae61", "#1a9641", "#d7191c"]
    x       = np.arange(len(order))
    width   = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    fig, ax = plt.subplots(figsize=(14, 5))
    df = df.set_index("appliance").loc[order]

    for offset, metric, color in zip(offsets, metrics, colors):
        vals = df[metric].values.astype(float)
        bars = ax.bar(x + offset * width, vals, width,
                      color=color, alpha=0.85, label=metric, edgecolor="white")
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=6.5)

    ax.axvline(x=len(order) - 1.5, color="grey", linestyle=":", linewidth=1.2, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylabel("Score")
    ax.set_title("Held-out Test — Per Appliance Metrics")
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    out = os.path.join(out_dir, "heldout_summary.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_heldout_complexity(perf_path, out_dir="~/thesis/results/finale/"):
    """
    3 subplots (one per complexity class: negative / simple / complex).
    Each subplot: one stacked bar per appliance, stacked by TP/TN/FP/FN as
    % of total samples for that appliance within that complexity class.
    """
    perf_path = os.path.expanduser(perf_path)
    out_dir   = os.path.expanduser(out_dir)

    df = pd.read_csv(perf_path)
    df = df[df["complexity"] != "TOTAL"]

    appliances   = ["kettle", "toaster", "microwave", "dishwasher", "fridge", "washing_machine"]
    complexities = ["negative", "simple", "complex"]
    outcome_colors = {"TP": "#1a9641", "TN": "#4393c3", "FP": "#fdae61", "FN": "#d7191c"}
    outcomes = ["TP", "TN", "FP", "FN"]

    x     = np.arange(len(appliances))
    width = 0.55

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle("Held-out — TP/TN/FP/FN breakdown by complexity class (% within class per appliance)", fontsize=12)

    for ax, comp in zip(axes, complexities):
        sub = df[df["complexity"] == comp].set_index("appliance")
        # denominator: total samples for this appliance in this complexity class
        totals = sub[outcomes].sum(axis=1)

        bottoms = np.zeros(len(appliances))
        for outcome in outcomes:
            vals = np.array([
                100 * sub.loc[a, outcome] / totals[a] if a in sub.index and totals[a] > 0 else 0.0
                for a in appliances
            ])
            ax.bar(x, vals, width, bottom=bottoms,
                   color=outcome_colors[outcome], alpha=0.88,
                   label=outcome, edgecolor="white")
            # label segments >8% to avoid clutter
            for xi, (v, b) in enumerate(zip(vals, bottoms)):
                if v > 8:
                    ax.text(xi, b + v / 2, f"{v:.0f}%",
                            ha="center", va="center", fontsize=7, color="white", fontweight="bold")
            bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels(appliances, rotation=25, ha="right", fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of segments in class" if ax is axes[0] else "")
        ax.set_title(comp.capitalize(), fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        if ax is axes[-1]:
            ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    out = os.path.join(out_dir, "heldout_complexity.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_heldout_confidence(conf_path, out_dir="~/thesis/results/finale/"):
    """
    1×2 subplots: FP and FN confidence breakdown (borderline/moderate/misfire %) only.
    """
    conf_path = os.path.expanduser(conf_path)
    out_dir   = os.path.expanduser(out_dir)

    df = pd.read_csv(conf_path)

    appliances = ["kettle", "toaster", "microwave", "dishwasher", "fridge", "washing_machine"]
    band_colors = {"borderline": "#92c5de", "moderate": "#f4a582", "misfire": "#d6604d"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Held-out — Error Confidence Breakdown", fontsize=13)

    for col_idx, etype in enumerate(["FP", "FN"]):
        sub = df[df["error_type"] == etype].set_index("appliance")

        ax = axes[col_idx]
        bottoms = np.zeros(len(appliances))
        x = np.arange(len(appliances))
        for band in ["borderline", "moderate", "misfire"]:
            pct_col = f"{band}_pct"
            vals = np.array([sub.loc[a, pct_col] if a in sub.index else 0.0
                             for a in appliances])
            ax.bar(x, vals, bottom=bottoms, color=band_colors[band],
                   alpha=0.85, label=band, edgecolor="white")
            bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels(appliances, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of errors")
        ax.set_title(f"{etype} — Confidence Breakdown", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    out = os.path.join(out_dir, "heldout_confidence.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


plot_heldout_summary(
    "~/thesis/results/finale/pool_9_1_stride_3_kernel_5_heldout_failure_analysis_s1_performance.csv",
    "~/thesis/results/finale/pool_9_1_stride_3_kernel_5_heldout_failure_analysis.csv",
)

plot_heldout_complexity(
    "~/thesis/results/finale/pool_9_1_stride_3_kernel_5_heldout_failure_analysis_s1_performance.csv",
)

plot_heldout_confidence(
    "~/thesis/results/finale/pool_9_1_stride_3_kernel_5_heldout_failure_analysis_s2_confidence.csv",
)