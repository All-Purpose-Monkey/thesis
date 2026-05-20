import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

def compute_metrics(
    preds,
    targets,
    appliances
):

    results = {}
    preds_bin = (preds > 0.5).astype(int)

    for i, app in enumerate(appliances):
        results[f"{app}_accuracy"] = roc_auc_score(
            targets[:, i],
            preds[:, i]
        )
        results[f"{app}_precision"] = precision_score(
            targets[:, i],
            preds_bin[:, i],
            zero_division=0
        )
        results[f"{app}_recall"] = recall_score(
            targets[:, i],
            preds_bin[:, i],
            zero_division=0
        )
        results[f"{app}_f1"] = f1_score(
            targets[:, i],
            preds_bin[:, i],
            zero_division=0
        )
    return results