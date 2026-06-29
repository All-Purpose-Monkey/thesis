from tqdm import tqdm
import numpy as np
import torch
import copy
from utils.metrics import compute_metrics

def train_downstream(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    appliances,
    epochs,
    thresholds=None,
    early_stopping_patience=3,
    scheduler=None
):

    history = []
    best_f1 = -np.inf
    patience_counter = 0
    best_loss = np.inf
    for epoch in range(epochs):

        # ----------------------------
        # TRAIN
        # -----------------------------

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
        # -----------------------------
        # VALIDATION
        # -----------------------------
        metrics = evaluate_epoch(
            model,
            val_loader,
            device,
            appliances,
            thresholds=thresholds,
            loss_fn=loss_fn
        )

        f1_scores = [
            metrics[f"{app}_f1"]
            for app in appliances
        ]
        current_f1 = np.nanmean(f1_scores)
        current_loss = total_loss / len(train_loader)
        metrics["mean_f1"] = current_f1
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = current_loss
        history.append(metrics)
        print(metrics)

         # Mean F1 across appliances
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(current_loss)
            else:
                scheduler.step()
        # -----------------------------
        # EARLY STOPPING
        # -----------------------------
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_state = copy.deepcopy(model.state_dict())  # save best weights
            
        if current_loss < best_loss:
            patience_counter = 0
            best_loss = current_loss
            print(
                f"New lowest loss: "
                f"{current_loss:.4f}"
                f"average f1: {current_f1:.4f}"
            )
        else:
            patience_counter += 1
            print(
                f"No improvement in loss fn"
                f"for {patience_counter} epoch(s)"
            )
            if patience_counter >= early_stopping_patience:
                print(
                    f"Early stopping triggered "
                    f"at epoch {epoch + 1}"
                )
                model.load_state_dict(best_state)  # restore best weights
                break
    
    model.load_state_dict(best_state)  # fail safe cuz my previous runs didnt hit eaarly stopping
    return history


def evaluate_epoch(
    model,
    loader,
    device,
    appliances,
    thresholds=None,
    loss_fn=None
):

    model.eval()
    all_preds = []
    all_targets = []
    total_val_loss=0

    with torch.no_grad():

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            batch_y=batch_y.to(device)
            if loss_fn is not None:
                total_val_loss += loss_fn(logits, batch_y).item()
            probs = torch.sigmoid(logits)
            all_preds.append(
                probs.cpu().numpy()
            )
            all_targets.append(
                batch_y.cpu().numpy()
            )

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    result = compute_metrics(all_preds, all_targets, appliances, thresholds=thresholds)
    if loss_fn is not None:
        result["val_loss"] = total_val_loss / len(loader)
    return result