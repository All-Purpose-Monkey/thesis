import torch.nn as nn
import torch.nn.functional as F
import torch

cos = nn.CosineSimilarity(dim=1, eps=1e-6)

def negative_cosine_similarity(p, z):

    return -(cos(p, z.detach()).mean())


def classification_loss(pos_weights=None, focal=False, gamma=2.0):

    if focal:
        def loss_fn(inputs, targets):
            return focal_loss(inputs, targets.float(), gamma=gamma, pos_weights=pos_weights)
        return loss_fn
    else:
        return nn.BCEWithLogitsLoss(pos_weight=pos_weights)

def focal_loss(inputs, targets, gamma=2.0, pos_weights=None):
    """
    Multilabel focal loss. gamma=2.0 is the standard starting point.
    pos_weights: same tensor as passed to BCE, handles class imbalance.
    """
    bce = F.binary_cross_entropy_with_logits(
        inputs, targets, pos_weight=pos_weights, reduction="none"
    )
    probs = torch.sigmoid(inputs)
    p_t = probs * targets + (1 - probs) * (1 - targets)  # prob of correct class
    focal_weight = (1 - p_t) ** gamma
    loss = focal_weight * bce
    return loss.mean()