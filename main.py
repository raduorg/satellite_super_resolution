
import os
import random
from datetime import datetime
import shutil
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
from models.srresnet import SRResNetLite
from models.ae import SRAutoEncoderLite
from models.mlp_mixer import MLPMixerSR
from models.edsr import EDSR
from utils.dataset import SuperResDataset

SEED        = 123
BATCH_SIZE  = 32
LR          = 2e-4
BASE_LR = 2e-4          # start LR
MIN_LR  = 1e-6          # final LR at the end of training
WEIGHT_DECAY = 1e-4

EPOCHS      = 100
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
HR_SIZE     = 64   # EuroSAT original image size
LR_SIZE     = 16   # Downscaled LR image size
NUM_WORKERS = 4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


train_loader = DataLoader(
    SuperResDataset('train.csv'),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)
val_loader = DataLoader(
    SuperResDataset('val.csv'),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)





def psnr(pred, gt):
    mse = F.mse_loss(pred, gt)
    return 10 * torch.log10(1.0 / mse)


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['cnn', 'ae', 'mlp', 'edsr'], default='cnn', help='Model to use: cnn, ae, mlp, or edsr')

# MLP Mixer arguments
parser.add_argument('--mlp_patch_size', type=int, default=4, help='MLP Mixer: patch size')
parser.add_argument('--mlp_embed_dim', type=int, default=128, help='MLP Mixer: embedding dimension')
parser.add_argument('--mlp_n_layers', type=int, default=6, help='MLP Mixer: number of mixer layers')
parser.add_argument('--mlp_token_mlp_dim', type=int, default=256, help='MLP Mixer: token mixing MLP hidden dimension')
parser.add_argument('--mlp_channel_mlp_dim', type=int, default=512, help='MLP Mixer: channel mixing MLP hidden dimension')
parser.add_argument('--mlp_p_drop', type=float, default=0.0, help='MLP Mixer: dropout probability')

# EDSR arguments
parser.add_argument('--edsr_n_feats', type=int, default=64, help='EDSR: number of feature channels')
parser.add_argument('--edsr_n_resblocks', type=int, default=16, help='EDSR: number of residual blocks')
parser.add_argument('--edsr_res_scale', type=float, default=1.0, help='EDSR: residual scaling factor')

args = parser.parse_args()

# Model selection
if args.model == 'cnn':
    model = SRResNetLite().to(DEVICE)
    model_name = 'cnn'
elif args.model == 'ae':
    model = SRAutoEncoderLite().to(DEVICE)
    model_name = 'ae'
elif args.model == 'mlp':
    model = MLPMixerSR(
        patch_size=args.mlp_patch_size,
        scale=HR_SIZE // LR_SIZE,
        img_size=(LR_SIZE, LR_SIZE),
        in_chans=3,
        embed_dim=args.mlp_embed_dim,
        n_layers=args.mlp_n_layers,
        token_mlp_dim=args.mlp_token_mlp_dim,
        channel_mlp_dim=args.mlp_channel_mlp_dim,
        p_drop=args.mlp_p_drop
    ).to(DEVICE)
    model_name = 'mlp'
elif args.model == 'edsr':
    model = EDSR(
        n_feats=args.edsr_n_feats,
        n_resblocks=args.edsr_n_resblocks,
        res_scale=args.edsr_res_scale,
        scale=HR_SIZE // LR_SIZE
    ).to(DEVICE)
    model_name = 'edsr'
else:
    raise ValueError('Unknown model type')

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=BASE_LR,
    betas=(0.9, 0.99),          # default Adam betas
    weight_decay=WEIGHT_DECAY   # L2 regularisation
)

# Cosine decay from BASE_LR → MIN_LR over all epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,               # number of epochs until cosine reaches minimum
    eta_min=MIN_LR
)
criterion = nn.L1Loss()
best_psnr = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for lr_img, hr_img in tqdm(train_loader, desc=f'Train {epoch}/{EPOCHS}', leave=False):
        lr_img = lr_img.to(DEVICE)
        hr_img = hr_img.to(DEVICE)
        optimizer.zero_grad()
        sr = model(lr_img)
        loss = criterion(sr, hr_img)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * lr_img.size(0)
    scheduler.step()

    model.eval()
    val_psnr_vals = []
    with torch.no_grad():
        for lr_img, hr_img in val_loader:
            lr_img = lr_img.to(DEVICE)
            hr_img = hr_img.to(DEVICE)
            sr = model(lr_img)
            val_psnr_vals.append(psnr(sr, hr_img).item())
    val_psnr = np.mean(val_psnr_vals)
    print(f'Epoch {epoch:02d} | Train Loss: {train_loss/len(train_loader.dataset):.4f} | Val PSNR: {val_psnr:.2f}')

    if val_psnr > best_psnr:
        best_psnr = val_psnr

    torch.save(model.state_dict(), f'best_{model_name}.pth')
    print(f'Saved new best model: best_{model_name}.pth')


model.load_state_dict(torch.load(f'best_{model_name}.pth', map_location=DEVICE))
model.eval()

test_loader = DataLoader(
    SuperResDataset('test.csv', with_target=False),
    batch_size=1,
    shuffle=False
)

# Create output directory for SR images
output_dir = 'output_sr'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

from PIL import Image as PILImage

print(f'Running inference on {len(test_loader)} test images...')
with torch.no_grad():
    for lr_img, filename in tqdm(test_loader, desc='Inference'):
        sr = model(lr_img.to(DEVICE)).cpu().squeeze(0).numpy()
        sr_uint8 = (sr * 255.0).round().clip(0, 255).astype(np.uint8)
        # Convert CHW to HWC for PIL
        sr_hwc = sr_uint8.transpose(1, 2, 0)
        # Save as JPG
        output_path = os.path.join(output_dir, filename[0].replace('.jpg', '_sr.jpg'))
        PILImage.fromarray(sr_hwc).save(output_path, 'JPEG', quality=95)

print(f'Finished. SR images saved to {output_dir}/')
print('Finished. Submission file created.')