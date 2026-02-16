import os
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure
import argparse
from models.mlp_mixer import MLPMixerSR
from models.edsr import EDSR
from models.bicubic import BicubicSR
from utils.dataset import SRDataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HR_SIZE = 64


def geometric_ensemble(model, lr_img, device):
    """8x TTA: rotations + flips"""
    x = lr_img.to(device)
    x_aug = []
    
    for k in range(4):
        x_aug.append(torch.rot90(x, k, dims=[2, 3]))
        
    x_flip = torch.flip(x, dims=[3])
    for k in range(4):
        x_aug.append(torch.rot90(x_flip, k, dims=[2, 3]))

    batch_aug = torch.cat(x_aug, dim=0)
    with torch.no_grad():
        sr_batch = model(batch_aug)
    sr_out = []
    
    for k in range(4):
        sr_out.append(torch.rot90(sr_batch[k:k+1], -k, dims=[2, 3]))
        
    for k in range(4):
        idx = 4 + k
        unrot = torch.rot90(sr_batch[idx:idx+1], -k, dims=[2, 3])
        sr_out.append(torch.flip(unrot, dims=[3]))

    sr_mean = torch.cat(sr_out, dim=0).mean(dim=0, keepdim=True)

    return sr_mean


def psnr(pred, gt):
    mse = F.mse_loss(pred, gt)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(1.0 / mse)


def load_checkpoint(checkpoint_path, device):
    if checkpoint_path.endswith('.pth'):
        pth_path = checkpoint_path
        json_path = checkpoint_path.replace('.pth', '_config.json')
    else:
        pth_path = f"{checkpoint_path}.pth"
        json_path = f"{checkpoint_path}_config.json"
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Checkpoint not found: {pth_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Config not found: {json_path}")
    with open(json_path, 'r') as f:
        config = json.load(f)
    model_type = config['model_type']
    arch = config['architecture']
    if model_type == 'edsr':
        model = EDSR(
            n_feats=arch.get('n_feats', 64),
            n_resblocks=arch.get('n_resblocks', 16),
            res_scale=arch.get('res_scale', 1.0)
        )
    elif model_type == 'mlp':
        model = MLPMixerSR(
            img_size=arch.get('img_size', 16),
            patch_size=arch.get('patch_size', 2),
            in_chans=arch.get('in_chans', 3),
            num_classes=arch.get('num_classes', 3),
            embed_dim=arch.get('embed_dim', 128),
            depth=arch.get('depth', 6),
            mlp_ratio=arch.get('mlp_ratio', 4.0)
        )
    else:
        raise ValueError(f"Unknown model type:'{model_type}'. supported models: edsr, mlp")
    model = model.to(device)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    return model, config


def main():
    import warnings
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--model', type=str, required=True, choices=['edsr', 'mlp', 'bicubic'])
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--set', type=str, required=True, choices=['test', 'val'])
    args = parser.parse_args()

    if args.model == 'bicubic':
        model = BicubicSR(scale=4).to(DEVICE)
        config = {
            'model_type': 'bicubic',
            'architecture': {'scale': 4},
            'training': 'N/A (no training)'
        }
    else:
        if args.checkpoint is None:
            print(f"Error: --checkpoint is required for model '{args.model}'")
            return 1
        
        checkpoint_path = args.checkpoint
        if os.path.sep not in checkpoint_path and not checkpoint_path.startswith('.'):
            checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_path)
        if checkpoint_path.endswith('.pth'):
            checkpoint_path = checkpoint_path[:-4]

        print(f"Loading checkpoint: {checkpoint_path}")
        try:
            model, config = load_checkpoint(checkpoint_path, DEVICE)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1

    model.eval()

    print(f"\nModel: {config['model_type'].upper()}")
    print(f"Architecture: {config['architecture']}")
    print(f"Training config: {config['training']}")
    if args.tta:
        print("TTA: Enabled (8x geometric ensemble)")

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

    def compute_metrics(sr, gt):
        if sr.dim() == 3:
            sr = sr.unsqueeze(0)
        if gt.dim() == 3:
            gt = gt.unsqueeze(0)
        mse = torch.mean((sr - gt) ** 2).item()
        psnr_val = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
        mse_val = mse
        ssim_val = ssim_metric(sr, gt).item()
        return psnr_val, ssim_val, mse_val

    csv_file = 'validation.csv' if args.set == 'val' else 'test.csv'
    set_name = 'VALIDATION' if args.set == 'val' else 'TEST'
    
    print(f"\n{set_name} MODE")
    data_loader = DataLoader(
        SRDataset(csv_file),
        batch_size=1,
        shuffle=False
    )
    print(f"Running {'8x TTA ' if args.tta else ''}inference on {len(data_loader)} {set_name.lower()} images...")
    
    psnr_vals, ssim_vals, mse_vals = [], [], []
    with torch.no_grad():
        for lr_img, hr_img in tqdm(data_loader, desc=set_name.title()):
            hr_img = hr_img.to(DEVICE)
            if args.tta:
                sr = geometric_ensemble(model, lr_img, DEVICE)
            else:
                sr = model(lr_img.to(DEVICE))
            p, s, m = compute_metrics(sr, hr_img)
            psnr_vals.append(p)
            ssim_vals.append(s)
            mse_vals.append(m)
    
    print(f"\n{set_name} SET RESULTS")
    print(f"Mean PSNR:  {np.mean(psnr_vals):.2f} dB")
    print(f"Mean SSIM:  {np.mean(ssim_vals):.4f}")
    print(f"Mean MSE:   {np.mean(mse_vals):.6f}")
    print(f"Std PSNR:   {np.std(psnr_vals):.2f} dB")
    print(f"Min PSNR:   {np.min(psnr_vals):.2f} dB")
    print(f"Max PSNR:   {np.max(psnr_vals):.2f} dB")
    return 0


if __name__ == '__main__':
    exit(main())
