import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from data.augmentation import view_1, view_2
from training.loss import negative_cosine_similarity
import numpy as np
import copy


def compute_hardness_weights(simsiam, dataset, device, batch_size=64):
    """
    Returns a weight per sample: lower cosine similarity between views = higher weight.
    """
    simsiam.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_sims = []

    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)

            # apply both views
            x_np = x.cpu().numpy()   # (B, 1, F, T)
            x1 = torch.tensor(
                np.stack([view_1(x_np[i]) for i in range(len(x_np))]),
                dtype=torch.float32
            ).to(device)
            x2 = torch.tensor(
                np.stack([view_2(x_np[i]) for i in range(len(x_np))]),
                dtype=torch.float32
            ).to(device)

            z1, p1 = simsiam(x1)   # adjust to match your simsiam forward signature
            z2, p2 = simsiam(x2)

            # cosine similarity per sample, averaged across the two directions
            sim = 0.5 * (
                F.cosine_similarity(p1, z2.detach(), dim=1) +
                F.cosine_similarity(p2, z1.detach(), dim=1)
            )                       # shape (batch,), range roughly [-1, 1]
            all_sims.append(sim.cpu())

    all_sims = torch.cat(all_sims)          # (N,)

    # invert: low sim = high weight; clamp to avoid zero weights
    weights = 1.0 - all_sims.clamp(-1, 1)  # now in [0, 2]
    weights = weights.clamp(min=0.05)       # floor so easy examples still appear
    return weights


def train_ssl(simsiam, dataset, optimizer, device, epochs, scheduler, patience=10):
    """
    SSL training loop with hard example mining refreshed each epoch.
    """
    best_loss = np.inf
    patience_counter = 0
    best_state = None  # for weight restore

    loader = None  # will be defined in first epoch after computing weights

    for epoch in range(epochs):
        if epoch % 10 == 0 or loader is None:

        # recompute weights every epoch (or every N epochs to save compute)
            weights = compute_hardness_weights(simsiam, dataset, device)
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            loader  = DataLoader(dataset, batch_size=256, num_workers=4, pin_memory=True, sampler=sampler)

        simsiam.train()
        total_loss = 0.0

        for (x,) in loader:
            x_np = x.cpu().numpy()
            x1 = torch.tensor(
                np.stack([view_1(x_np[i]) for i in range(len(x_np))]),
                dtype=torch.float32
            ).to(device)
            x2 = torch.tensor(
                np.stack([view_2(x_np[i]) for i in range(len(x_np))]),
                dtype=torch.float32
            ).to(device)

            z1, p1 = simsiam(x1)
            z2, p2 = simsiam(x2)
            loss = (
                negative_cosine_similarity(p1, z2) / 2 +
                negative_cosine_similarity(p2, z1) / 2
            )   

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step()
        print(f"[SSL] Epoch {epoch+1} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_state = copy.deepcopy(simsiam.state_dict())  # save best weights
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[SSL] Early stopping at epoch {epoch+1}")
                simsiam.load_state_dict(best_state)  # restore best
                break

