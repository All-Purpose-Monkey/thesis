from tqdm import tqdm
import numpy as np
import torch

from utils.metrics import compute_metrics

def train_downstream(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    appliances,
    epochs
):

    history = []
    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for batch_x, batch_y in tqdm(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            preds = model(batch_x)
            loss = loss_fn(preds, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        metrics = evaluate_epoch(
            model,
            val_loader,
            device,
            appliances
        )

        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = total_loss / len(train_loader)
        history.append(metrics)
        print(metrics)
    return history


def evaluate_epoch(
    model,
    loader,
    device,
    appliances
):

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.sigmoid(logits)
            all_preds.append(
                probs.cpu().numpy()
            )
            all_targets.append(
                batch_y.numpy()
            )

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    return compute_metrics(
        all_preds,
        all_targets,
        appliances
    )