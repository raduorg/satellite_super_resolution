import torch
import torch.nn as nn
import math

class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, bias=True, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2, bias=bias))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class EDSR(nn.Module):
    def __init__(self, n_feats=64, n_resblocks=16, res_scale=1.0, scale=4):
        super(EDSR, self).__init__()
        
        kernel_size = 3 
        #head
        self.head = nn.Conv2d(3, n_feats, kernel_size, padding=kernel_size//2)
        #body
        self.body = nn.Sequential(*[
            ResBlock(n_feats, kernel_size, res_scale=res_scale) for _ in range(n_resblocks)
        ])
        #conv after blocks
        self.body_tail = nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2)
        #upsampler
        self.upsampler = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * (scale ** 2), kernel_size, padding=kernel_size//2),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_feats, 3, kernel_size, padding=kernel_size//2)
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res = self.body_tail(res)
        res += x
        x = self.upsampler(res)
        return x
