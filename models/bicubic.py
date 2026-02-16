import torch
import torch.nn as nn
import torch.nn.functional as F


class BicubicSR(nn.Module):
    def __init__(self, scale: int = 4):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x,
            scale_factor=self.scale,
            mode='bicubic',
            align_corners=False,
            antialias=True
        )
