import math
import torch
import torch.nn as nn


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """ Turn an image into a sequence of flattened, linearly-projected patches. output: x_seq (B , N , C) and spatial size (h_p , w_p) """
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        ps = self.patch_size
        assert H % ps == 0 and W % ps == 0,  \
            f'Image size must be divisible by patch size {ps}'
        x = self.proj(x)                     # (B , embed , H/ps , W/ps)
        h_p, w_p = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)     # (B , N , embed)
        return x, h_p, w_p


class MLP(nn.Module):
    """ simple 2-layer MLP as used in the mixer blocks """
    def __init__(self, in_dim, hidden_dim, p_drop=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden_dim, in_dim),
            nn.Dropout(p_drop)
        )

    def forward(self, x):
        return self.net(x)


class MixerLayer(nn.Module):
    """ One Mixer layer = token mixing MLP + channel mixing MLP input shape : (B , N_tokens , C_embed) """
    def __init__(self, num_tokens: int, embed_dim: int, token_mlp_dim: int, channel_mlp_dim: int, p_drop: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.token_mlp = MLP(num_tokens, token_mlp_dim, p_drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.channel_mlp = MLP(embed_dim, channel_mlp_dim, p_drop)

    def forward(self, x):
        y = self.norm1(x)
        # --- token mixing (operate over N dimension) ---
        y = y.transpose(1, 2)          # (B , C , N)
        y = self.token_mlp(y)
        y = y.transpose(1, 2)          # back to (B , N , C)
        x = x + y                      # residual
        # --- channel mixing (operate over C dimension) ---
        y = self.norm2(x)
        y = self.channel_mlp(y)
        x = x + y
        return x


# ------------------------------------------------------------------
# MAIN NETWORK
# ------------------------------------------------------------------
class MLPMixerSR(nn.Module):
    """ Super-Resolution network based purely on MLP-Mixer idea. 1. Split LR image in non-overlapping patches 2. Run a stack of Mixer layers (token+channel mixing) 3. Reshape tokens back to feature map 4. Learned up-sampling (Conv -> PixelShuffle -> Conv) 5. Output HR RGB image ----------------------------------------------------------------- Parameters ---------- scale : integer up-scaling factor (2,3,4…) img_size : tuple(h,w) of the LOW-RES input size (needed so that token-mixing MLPs have fixed size) patch_size : size of square patches (e.g. 4 or 8) in_chans : usually 3 embed_dim : hidden dimension of each token n_layers : how many Mixer layers token_mlp_dim : hidden dim of token-mixing MLP channel_mlp_dim: hidden dim of channel-mixing MLP """
    def __init__(self, patch_size: int = 4, scale: int = 4, img_size: tuple = (16, 16), in_chans: int = 3, embed_dim: int = 256, n_layers: int = 8, token_mlp_dim: int = 512, channel_mlp_dim: int = 1024, p_drop: float = 0.):
        super().__init__()
        ih, iw = img_size
        assert ih % patch_size == 0 and iw % patch_size == 0, "img_size must be divisible by patch_size"
        self.h_p = ih // patch_size
        self.w_p = iw // patch_size
        num_tokens = self.h_p * self.w_p

        self.patch_size = patch_size
        self.scale = scale
        self.final_up = patch_size * scale  # total enlargement

        # 1. patch embedding
        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size)

        # 2. mixer stack
        self.mixer = nn.Sequential(*[
            MixerLayer(num_tokens, embed_dim, token_mlp_dim, channel_mlp_dim, p_drop) for _ in range(n_layers)
        ])

        # 3. learned up-sampler (Conv -> PixelShuffle)
        self.upsampler = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * (self.final_up ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(self.final_up),
            nn.Conv2d(embed_dim, in_chans, kernel_size=3, padding=1)
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        # same simple init rule for every Linear/Conv
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x):
        """ x : (B , 3 , H_in , W_in) LOW-RES image with size given at init. """
        B = x.size(0)
        tokens, hp, wp = self.patch_embed(x)              # (B , N , C)
        assert hp == self.h_p and wp == self.w_p, \
            "Input size at inference must match img_size given in constructor"

        tokens = self.mixer(tokens)                       # (B , N , C)

        # Reshape tokens back to a (B , C , h_p , w_p) map
        tokens = tokens.transpose(1, 2)                   # (B , C , N)
        feat = tokens.reshape(B, -1, self.h_p, self.w_p)  # (B , C , h_p , w_p)

        # Learned up-sampling + RGB reconstruction
        out = self.upsampler(feat)                        # (B , 3 , H*scale , W*scale)
        return out


# ------------------------------------------------------------------
# simple check
# ------------------------------------------------------------------
if __name__ == '__main__':
    """ Example : take a 48x48 LR image, upscale x4 → 192x192 """
    model = MLPMixerSR(scale=4,
                       img_size=(16, 16),   # low-res size
                       patch_size=4,
                       embed_dim=256,
                       n_layers=6)

    lr = torch.randn(1, 3, 16, 16)   # B,C,H,W
    sr = model(lr)
    print('LR shape:', lr.shape)
    print('SR shape:', sr.shape)
    # -> SR shape: torch.Size([1, 3, 192, 192])