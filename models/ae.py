import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(cin, cout, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, p),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True)
    )

class PixelShuffleBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch * 4, 3, 1, 1)
        self.ps   = nn.PixelShuffle(2)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))


class SRAutoEncoderLite(nn.Module):
    def __init__(self, nf: int = 64):
        super().__init__()
        self.enc = nn.Sequential(
            conv_bn_relu(3,   nf,   3, 1, 1),
            conv_bn_relu(nf,  nf,   3, 1, 1),
            conv_bn_relu(nf,  nf*2, 3, 1, 1),
            conv_bn_relu(nf*2, nf*2, 3, 1, 1)
        )
        #decoder
        self.dec = nn.Sequential(
            PixelShuffleBlock(nf*2),
            conv_bn_relu(nf*2, nf, 3, 1, 1),  

            PixelShuffleBlock(nf),
            conv_bn_relu(nf, nf // 2, 3, 1, 1),
            #rgb layer
            nn.Conv2d(nf // 2, 3, 3, 1, 1)
        )
        self.register_buffer('mean', torch.tensor([0.0]))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        lr_up = F.interpolate(x, scale_factor=4, mode='nearest')
        z     = self.enc(x)
        sr    = self.dec(z)
        return sr + lr_up * 0.1 