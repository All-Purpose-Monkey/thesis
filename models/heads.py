import torch
import torch.nn as nn


class MultiHeadClassifier_big(nn.Module):

    def __init__(self, backbone, appliances):

        super().__init__()

        self.backbone = backbone

        self.shared = nn.Sequential(
            nn.Linear(9216, 2048),  # 128*12*6
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU()
        )

        self.heads = nn.ModuleDict({
            app: nn.Sequential(
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )
            for app in appliances
        })

    def forward(self, x):

        feat = self.backbone(x)
        shared = self.shared(feat)

        outputs = []
        for app in self.heads:
            outputs.append(self.heads[app](shared))

        return torch.cat(outputs, dim=1)
    
class MultiHeadClassifier_mini(nn.Module):

    def __init__(self, backbone, appliances, k_num=16):
        super().__init__()
        self.backbone = backbone
        self.k_num=k_num
        self.shared = nn.Sequential(
            nn.Linear(backbone.output_dim(), k_num*2),
            nn.LayerNorm(k_num*2),
            nn.ReLU(),
        )

        self.heads = nn.ModuleDict({
            app: nn.Sequential(
                nn.Linear(k_num*2, k_num),
                nn.LayerNorm(k_num),
                nn.ReLU(),
                nn.Linear(k_num, 1)
            )
            for app in appliances
        })

    def forward(self, x):
        feat = self.backbone(x)
        shared = self.shared(feat)
        outputs = []
        for app in self.heads:
            outputs.append(self.heads[app](shared))
        return torch.cat(outputs, dim=1)