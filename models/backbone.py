import torch
import torch.nn as nn


class CNNBackbone(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
# compression blocks
            # (1, 576, 192) -> (32, 288, 96)
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # (32, 288, 96) -> (64, 144, 48)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            
            # (64, 144, 48) -> (128, 72, 24)
            nn.Conv2d(64, 128, kernel_size=3, stride=2,
                      padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
# dilated blocks — time dilation only, freq dilation=1 as merging frequency band info reduced appliance discriminability in early experiments
            # (128, 72, 24) -> (128, 72, 24)
            nn.Conv2d(128, 256, kernel_size=3, stride=1,
                      padding=(1, 1), dilation=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # (256, 72, 24) -> (512, 72, 24)
            nn.Conv2d(256, 512, kernel_size=3, stride=1,
                      padding=(1, 2), dilation=(1, 2)),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # (512, 144, 48) -> (512, 144, 48)
            nn.Conv2d(512, 512, kernel_size=3, stride=1,
                      padding=(1, 4), dilation=(1, 4)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
# channel suppression + global pooling for increasing invariance and reducing overfitting - MLP heads going whack sized without channel supression          
            # (512, 144, 48) -> (256, 144, 48)  RF_time = 63 >= 48 ✓
            nn.Conv2d(512, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # (256, 144, 48) -> (128, 144, 48)  RF_time = 63 >= 48 ✓
            nn.Conv2d(256, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((12, 6)) #final suppressions for creating rich embeddings
        )

    def forward(self, x):
        x = torch.nn.functional.pad(x, (0, 4, 0, 63))
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        return x

    def output_dim(self):
        # 128 * 12 * 6
        return 9216
    
class CNNmini(nn.Module):

    def __init__(self, k_num=16, pool_size=(1, 1), stride=2, padding=0, kernel_size=3, sym_pad=False):
        super().__init__()
        self.k_num = k_num
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.sym_pad = sym_pad

        self.encoder = nn.Sequential(
            # (1, 576, 192) -> (k_num, 288, 96)
            nn.Conv2d(1, k_num, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(k_num),
            nn.ReLU(),

            # (k_num, 288, 96) -> (k_num*2, 144, 48)
            nn.Conv2d(k_num, k_num * 2, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(k_num * 2),
            nn.ReLU(),

            # (k_num*2, 144, 48) -> (k_num*4, 72, 24)
            nn.Conv2d(k_num * 2, k_num * 4, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(k_num * 4),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(pool_size)
        )

    def forward(self, x):
        if self.sym_pad == True:
            x = torch.nn.functional.pad(x, (0, 4, 0, 63))
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        return x

    def output_dim(self):
        return self.k_num * 4 * self.pool_size[0] * self.pool_size[1]
    