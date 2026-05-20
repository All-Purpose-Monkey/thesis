import torch
import torch.nn as nn

class MultiHeadClassifier(nn.Module):

    def __init__(self, backbone, appliances):

        super().__init__()

        self.backbone = backbone

        self.shared = nn.Sequential(
            nn.Linear(9216, 1024),   #256*6*6 = 9216
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.ReLU()
        )

        self.heads = nn.ModuleDict({
            app: nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )

            for app in appliances
        })

    def forward(self, x):

        feat = self.backbone(x)

        shared = self.shared(feat)

        outputs = []

        for app in self.heads:

            outputs.append(
                self.heads[app](shared)
            )

        return torch.cat(outputs, dim=1)