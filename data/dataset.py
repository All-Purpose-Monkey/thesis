import torch
from torch.utils.data import Dataset

class STFTDataset(Dataset):

    def __init__(self, X, y=None):

        self.X = X
        self.y = y

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        stft = torch.tensor(
            self.X[idx],
            dtype=torch.float32
        ).unsqueeze(0)

        if self.y is not None:

            label = torch.tensor(
                self.y[idx],
                dtype=torch.float32
            )

            return stft, label

        return stft