from tqdm import tqdm
import torch
from data.augmentation import time_mask
from data.augmentation import freq_mask

from training.loss import negative_cosine_similarity

def train_ssl(
    model,
    dataloader,
    optimizer,
    device,
    epochs,
    scheduler=None,
):

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for batch in tqdm(dataloader):

            batch = batch.to(device)

            view1 = batch.clone()
            view2 = batch.clone()

            for i in range(batch.shape[0]):

                view1[i] = time_mask(view1[i])
                view1[i] = freq_mask(view1[i])

                view2[i] = time_mask(view2[i])
                view2[i] = freq_mask(view2[i])

            z1, p1 = model(view1)
            z2, p2 = model(view2)

            loss = (
                negative_cosine_similarity(p1, z2)/2
                +
                negative_cosine_similarity(p2, z1)/2
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(
            f"SSL Epoch {epoch+1} "
            f"Loss: {total_loss/len(dataloader):.4f}"
        )