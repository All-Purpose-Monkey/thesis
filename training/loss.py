import torch.nn as nn

cos = nn.CosineSimilarity(dim=1, eps=1e-6)

def negative_cosine_similarity(p, z):

    return -(cos(p, z.detach()).mean())


def classification_loss(pos_weights=None):

    return nn.BCEWithLogitsLoss(pos_weight=pos_weights)