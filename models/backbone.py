import torch
import torch.nn as nn

class CNNBackbone(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(1, 32, 3, stride=2, padding=1), # 576,192 -> 288,96
            nn.BatchNorm2d(32),
            nn.ReLU(),
            

            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 288,96 -> 144,48
            nn.BatchNorm2d(64),
            nn.ReLU(),


            nn.Conv2d(64, 128, 3, stride=1, padding=1), # 144,48 -> 144,48 (keep spatial dimensions for more local features)
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2, 2), #for speed up and redundancy reduction 144,48 -> 72,24

            nn.Conv2d(128, 256, 3, stride=1, padding=1), # 72,24 -> 72,24
            nn.BatchNorm2d(256),
            nn.ReLU(),

            #nn.MaxPool2d(2, 2), #for speed up and redundancy reduction 72,24 -> 36,12

            nn.AdaptiveAvgPool2d((6,6)) # for fixed output size regardless of input size, 72,24 -> 6,6 gets a second's worth of temporal info and all freq info
        )

    def forward(self, x):
        x = torch.nn.functional.pad(x, (0, 4, 0, 63))  # (513,188) → (576,192) - need this for clean divisibles so gpu doesnt crash
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        return x