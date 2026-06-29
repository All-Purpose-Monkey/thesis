import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

def threshold_sweep(preds, targets, appliances, thresholds=None):
    """
    Sweep decision thresholds per appliance and return the best
    threshold + metrics for each, plus the full curve for plotting.

    Args:
        preds:       np.ndarray, shape (N, num_appliances), raw sigmoid outputs
        targets:     np.ndarray, shape (N, num_appliances), binary labels
        appliances:  list[str]
        thresholds:  array-like of floats to sweep, default 0.05..0.95

    Returns:
        best:   dict  { appliance: { "threshold", "f1", "precision", "recall" } }
        curves: dict  { appliance: { "thresholds", "f1s", "precisions", "recalls" } }
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)

    best = {}
    curves = {}

    for i, app in enumerate(appliances):
        f1s, precs, recs = [], [], []

        for t in thresholds:
            preds_bin = (preds[:, i] >= t).astype(int)
            f1s.append(f1_score(targets[:, i], preds_bin, zero_division=0))
            precs.append(precision_score(targets[:, i], preds_bin, zero_division=0))
            recs.append(recall_score(targets[:, i], preds_bin, zero_division=0))

        best_idx = int(np.argmax(f1s))
        best[app] = {
            "threshold":  round(float(thresholds[best_idx]), 3),
            "f1":         round(f1s[best_idx], 4),
            "precision":  round(precs[best_idx], 4),
            "recall":     round(recs[best_idx], 4),
        }
        curves[app] = {
            "thresholds": thresholds.tolist(),
            "f1s":        f1s,
            "precisions": precs,
            "recalls":    recs,
        }

    return best, curves


def compute_metrics(preds, targets, appliances, thresholds=None):
    """
    Compute per-appliance classification metrics.
 
    Args:
        preds:      (N, num_appliances) float array of sigmoid probabilities
        targets:    (N, num_appliances) binary int array of ground truth
        appliances: list of appliance names
        thresholds: dict {app: float} or None (defaults to 0.5 for all)
    """
    if thresholds is None:
        thresholds = {app: 0.5 for app in appliances}
 
    results = {}
 
    for i, app in enumerate(appliances):
 
        t = targets[:, i]
        p = preds[:, i]
 
        # per-appliance threshold applied to THIS appliance's column only
        thresh = thresholds.get(app, 0.5)
        p_bin  = (p > thresh).astype(int)
 
        # guard: if test split has no positives, all metrics undefined
        if t.sum() == 0:
            results[f"{app}_roc_auc_score"] = float("nan")
            results[f"{app}_precision"]     = float("nan")
            results[f"{app}_recall"]        = float("nan")
            results[f"{app}_f1"]            = float("nan")
            continue
 
        try:
            results[f"{app}_roc_auc_score"] = roc_auc_score(t, p)
        except Exception:
            results[f"{app}_roc_auc_score"] = float("nan")
 
        results[f"{app}_precision"] = precision_score(t, p_bin, zero_division=0)
        results[f"{app}_recall"]    = recall_score(t, p_bin, zero_division=0)
        results[f"{app}_f1"]        = f1_score(t, p_bin, zero_division=0)
 
    return results