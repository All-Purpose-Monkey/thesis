import torch
import torch.nn as nn

class SimSiam(nn.Module):

    def __init__(self, backbone, feat_dim=9216): #256*6*6 = 9216

        super().__init__()

        self.backbone = backbone

        self.projection = nn.Sequential(

            nn.Linear(feat_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256)
        )

        self.prediction = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )

    def forward(self, x):

        feat = self.backbone(x)
        z = self.projection(feat)
        p = self.prediction(z)
        return z, p