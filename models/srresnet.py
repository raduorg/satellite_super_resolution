import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, c, res_scale=1.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, 1, 1)
        )
        self.scale = res_scale
    def forward(self, x):
        return x + self.body(x) * self.scale

class UpsampleBlock(nn.Module):
    def __init__(self, c, scale=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c*scale*scale, 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class SRResNetLite(nn.Module):
    def __init__(self, n_feats=64, n_resblocks=8, res_scale=1.0, upscale=4):
        super().__init__()
        assert upscale == 4, "fixed sr scope"
        self.head = nn.Conv2d(3, n_feats, 3, 1, 1)
        self.body = nn.Sequential(
            *[ResidualBlock(n_feats, res_scale) for _ in range(n_resblocks)]
        )
        self.up   = nn.Sequential(UpsampleBlock(n_feats),
                                  UpsampleBlock(n_feats))
        self.tail = nn.Conv2d(n_feats, 3, 3, 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.up(x)
        x = self.tail(x)
        return x.clamp(0, 1)

def build_srresnet_lite(**kwargs):
    return SRResNetLite(**kwargs)
