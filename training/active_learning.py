import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from data.dataset import STFTDataset


def uncertainty_scores(classifier, X_pool, device, batch_size=64, num_workers=4, pin_memory=True):
    """
    Score each unlabelled sample by prediction entropy across appliances.
    Higher entropy = more uncertain = higher priority to label.

    Args:
        classifier: trained MultiHeadClassifier (eval mode)
        X_pool:     list of STFT arrays (unlabelled)
        device:     torch.device

    Returns:
        np.ndarray of shape (N,), entropy scores
    """
    pool_dataset = STFTDataset(X_pool)          # no labels
    pool_loader  = DataLoader(pool_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    all_probs = []
    classifier.eval()

    with torch.no_grad():
        for (X_batch,) in pool_loader:          # STFTDataset returns tuple even without y
            X_batch = X_batch.to(device)
            logits  = classifier(X_batch)
            probs   = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)   # (N, num_appliances)

    # Per-appliance binary entropy, averaged across appliances
    eps     = 1e-8
    entropy = -(
        all_probs       * np.log(all_probs       + eps) +
        (1 - all_probs) * np.log(1 - all_probs   + eps)
    )                                                   # (N, num_appliances)

    return entropy.mean(axis=1)                         # (N,)


def select_query_indices(scores, n_query):
    """
    Return indices of the n_query most uncertain samples.
    Use enumerate + sorted rather than argsort for clarity.
    """
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in ranked[:n_query]]


def active_learning_round(
    classifier,
    X_labelled, 
    y_labelled,
    X_pool,    
    y_pool_oracle,   # oracle = labels pool withheld
    device,
    n_query=50,
    batch_size=64,
    num_workers=4,
    pin_memory=True
):
    """
    One active learning round:
      1. Score the pool by uncertainty.
      2. Pick top-n_query samples.
      3. Move them from pool → labelled set.
      4. Return updated sets + which indices were queried.

    Args:
        y_pool_oracle: np.ndarray, shape (pool_size, num_appliances)
                       In a real AL loop these would come from a human annotator.
                       Here we simulate by revealing withheld labels.

    Returns:
        X_labelled_new, y_labelled_new, X_pool_new, y_pool_new, queried_indices
    """
    scores          = uncertainty_scores(classifier, X_pool, device, batch_size, num_workers, pin_memory)
    queried_indices = select_query_indices(scores, n_query)
    queried_set     = set(queried_indices)

    # Move queried samples to labelled set
    X_labelled_new = np.concatenate([X_labelled, np.abs([X_pool[i] for i in queried_indices])], axis=0)
    y_labelled_new = np.concatenate([
        y_labelled,
        y_pool_oracle[queried_indices]
    ], axis=0)

    # Remove queried samples from pool using enumerate + list comp
    X_pool_new     = [x for i, x in enumerate(X_pool)        if i not in queried_set]
    y_pool_new     = np.array([y for i, y in enumerate(y_pool_oracle) if i not in queried_set])

    print(f"[AL] Queried {len(queried_indices)} samples. "
          f"Labelled set: {len(X_labelled_new)}, Pool: {len(X_pool_new)}")

    return X_labelled_new, y_labelled_new, X_pool_new, y_pool_new, queried_indices