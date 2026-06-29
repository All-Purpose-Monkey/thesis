import torch
import torch.nn as nn
import torch.nn.functional as F


class SimSiam(nn.Module):

    def __init__(self, backbone, feat_dim=9216):  # 128*12*6

        super().__init__()

        self.backbone = backbone

        self.projection = nn.Sequential(
            nn.Linear(feat_dim, 2048), #x4 compression
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
        )

        self.prediction = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )

    def forward(self, x):
        feat = self.backbone(x)
        z = self.projection(feat)
        p = self.prediction(z)
        return z, p