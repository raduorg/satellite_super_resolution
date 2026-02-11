"""
Training Script for Super-Resolution Models
============================================
Train EDSR or MLPMixerSR with specified hyperparameters.
Saves checkpoint with config JSON for reproducibility.

Usage:
    python train.py --model edsr --epochs 100 --n-feats 64 --n-resblocks 16 --augmentation flip_rotate
    python train.py --model mlp --epochs 100 --mlp-embed-dim 256 --mlp-n-layers 8
"""

import os
import json
import random
from datetime import datetime
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse

from models.edsr import EDSR
from models.mlp_mixer import MLPMixerSR
from utils.dataset import SRDataset

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
SEED = 123
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HR_SIZE = 64
LR_SIZE = 16
NUM_WORKERS = 4
MIN_LR = 1e-6

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def charbonnier(pred, target, eps=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))

# ─────────────────────────────────────────────────────────────
# Augmentation Strategies
# ─────────────────────────────────────────────────────────────
AUGMENTATION_STRATEGIES = {
    'none': {},
    'flip_h': {'horizontal_flip': True},
    'flip_hv': {'horizontal_flip': True, 'vertical_flip': True},
    'flip_rotate': {'horizontal_flip': True, 'vertical_flip': True, 'rotate90': True},
    'full_geometric': {'horizontal_flip': True, 'vertical_flip': True, 'rotate90': True, 'color_jitter': True},
}


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────
def psnr(pred, gt):
    mse = F.mse_loss(pred, gt)
    return 10 * torch.log10(1.0 / mse)


# ─────────────────────────────────────────────────────────────
# Checkpoint Functions
# ─────────────────────────────────────────────────────────────
def get_checkpoint_name(model_type, config):
    """Generate descriptive checkpoint name"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if model_type == 'edsr':
        parts = [
            model_type,
            f"nf{config['n_feats']}",
            f"rb{config['n_resblocks']}",
            f"rs{config['res_scale']}",
            f"lr{config['lr']:.0e}",
            f"bs{config['batch_size']}",
            f"aug-{config['augmentation']}",
            timestamp
        ]
    elif model_type == 'mlp':
        parts = [
            model_type,
            f"ps{config['patch_size']}",
            f"ed{config['embed_dim']}",
            f"nl{config['n_layers']}",
            f"lr{config['lr']:.0e}",
            f"bs{config['batch_size']}",
            f"aug-{config['augmentation']}",
            timestamp
        ]
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return '_'.join(parts)


def save_checkpoint(model, model_type, config, checkpoint_dir, name, best_psnr, best_epoch):
    """Save model checkpoint and config JSON"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model weights
    pth_path = os.path.join(checkpoint_dir, f"{name}.pth")
    torch.save(model.state_dict(), pth_path)
    
    # Build architecture config
    if model_type == 'edsr':
        architecture = {
            'n_feats': config['n_feats'],
            'n_resblocks': config['n_resblocks'],
            'res_scale': config['res_scale']
        }
    elif model_type == 'mlp':
        architecture = {
            'patch_size': config['patch_size'],
            'embed_dim': config['embed_dim'],
            'n_layers': config['n_layers'],
            'token_mlp_dim': config['token_mlp_dim'],
            'channel_mlp_dim': config['channel_mlp_dim'],
            'p_drop': config['p_drop']
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Save config JSON
    meta = {
        'model_type': model_type,
        'architecture': architecture,
        'loss_function': config['loss_function'],
        'training': {
            'lr': config['lr'],
            'batch_size': config['batch_size'],
            'weight_decay': config['weight_decay'],
            'augmentation': config['augmentation'],
            'epochs': config['epochs']
        },
        'metrics': {
            'best_psnr': best_psnr,
            'best_epoch': best_epoch
        },
        'timestamp': datetime.now().isoformat()
    }
    
    json_path = os.path.join(checkpoint_dir, f"{name}_config.json")
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f'Saved checkpoint: {pth_path}')
    print(f'Saved config: {json_path}')
    return pth_path


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Super-Resolution models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model selection
    parser.add_argument('--model', choices=['edsr', 'mlp'], required=True,
                        help='Model type: edsr (EDSR) or mlp (MLPMixerSR)')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--loss', choices=['l1', 'charbonnier'], default='l1', help='Loss function to use')
    
    # Augmentation
    parser.add_argument('--augmentation', choices=list(AUGMENTATION_STRATEGIES.keys()),
                        default='none', help='Augmentation strategy')
    
    # EDSR-specific
    parser.add_argument('--n-feats', type=int, default=64, help='[EDSR] Number of features')
    parser.add_argument('--n-resblocks', type=int, default=8, help='[EDSR] Number of residual blocks')
    parser.add_argument('--res-scale', type=float, default=1.0, help='[EDSR] Residual scaling factor')
    
    # MLP Mixer-specific
    parser.add_argument('--mlp-patch-size', type=int, default=4, help='[MLP] Patch size for tokenization')
    parser.add_argument('--mlp-embed-dim', type=int, default=128, help='[MLP] Embedding dimension per token')
    parser.add_argument('--mlp-n-layers', type=int, default=6, help='[MLP] Number of Mixer layers')
    parser.add_argument('--mlp-token-mlp-dim', type=int, default=256, help='[MLP] Token-mixing MLP hidden dimension')
    parser.add_argument('--mlp-channel-mlp-dim', type=int, default=512, help='[MLP] Channel-mixing MLP hidden dimension')
    parser.add_argument('--mlp-p-drop', type=float, default=0.0, help='[MLP] Dropout probability')
    
    # Fine tuning
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to .pth checkpoint to continue from')
    parser.add_argument('--ft-epochs', type=int, default=20,
                        help='Extra epochs when --resume is used')
    parser.add_argument('--lr-ft', type=float, default=5e-4,
                        help='New LR for fine-tuning phase')
    parser.add_argument('--reset-optim', action='store_true',
                        help='Start a fresh optimiser even when resuming')
    
    # Output
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Build config dict
    config = {
        'lr': args.lr,
        'batch_size': args.batch_size,
        'weight_decay': args.weight_decay,
        'augmentation': args.augmentation,
        'epochs': args.epochs,
        'loss_function': args.loss
    }
    
    # Calculate scale factor
    scale = HR_SIZE // LR_SIZE
    
    if args.model == 'edsr':
        config.update({
            'n_feats': args.n_feats,
            'n_resblocks': args.n_resblocks,
            'res_scale': args.res_scale,
        })
        model = EDSR(
            n_feats=args.n_feats,
            n_resblocks=args.n_resblocks,
            res_scale=args.res_scale,
            scale=scale
        ).to(DEVICE)
    elif args.model == 'mlp':
        config.update({
            'patch_size': args.mlp_patch_size,
            'embed_dim': args.mlp_embed_dim,
            'n_layers': args.mlp_n_layers,
            'token_mlp_dim': args.mlp_token_mlp_dim,
            'channel_mlp_dim': args.mlp_channel_mlp_dim,
            'p_drop': args.mlp_p_drop,
        })
        model = MLPMixerSR(
            patch_size=args.mlp_patch_size,
            scale=scale,
            img_size=(LR_SIZE, LR_SIZE),
            in_chans=3,
            embed_dim=args.mlp_embed_dim,
            n_layers=args.mlp_n_layers,
            token_mlp_dim=args.mlp_token_mlp_dim,
            channel_mlp_dim=args.mlp_channel_mlp_dim,
            p_drop=args.mlp_p_drop
        ).to(DEVICE)
    else:
        raise ValueError(f"Unknown model type: '{args.model}'. Supported models: edsr, mlp")
    
    print(f"{'='*60}")
    print(f"Training {args.model.upper()}")
    print(f"{'='*60}")
    print(f"Config: {config}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}\n")
    
    # Data loaders
    augmentation = AUGMENTATION_STRATEGIES[args.augmentation]
    train_loader = DataLoader(
        SRDataset('train.csv', root_dir=None, augmentation=augmentation),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        SRDataset('validation.csv', root_dir=None),  # No augmentation for validation
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # ------------------------------------------------------------
    # 1. Load checkpoint (weights only) if requested
    # ------------------------------------------------------------
    resume = args.resume is not None
    want_reset = args.reset_optim
    if resume:
        ckpt = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)

    # ------------------------------------------------------------
    # 2. Decide starting LR and epoch budget
    # ------------------------------------------------------------
    lr_start = args.lr_ft if (resume and args.lr_ft is not None) else args.lr
    if resume and args.ft_epochs:
        args.epochs = args.ft_epochs

    # ------------------------------------------------------------
    # 3. Create optimiser fresh
    # ------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=lr_start,
                                  betas=(0.9, 0.99),
                                  weight_decay=args.weight_decay)

    # ------------------------------------------------------------
    # 4. Optionally load old optimiser state and keep moments
    # ------------------------------------------------------------
    if resume and not want_reset and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
        # after loading, override LR in case lr_ft differs
        for g in optimizer.param_groups:
            g['lr'] = lr_start

    # ------------------------------------------------------------
    # 5. Update config dict so the banner and JSON are correct
    # ------------------------------------------------------------
    config['lr']     = lr_start
    config['epochs'] = args.epochs

    # ------------------------------------------------------------
    # 6. Scheduler now that optimiser is final
    # ------------------------------------------------------------
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=MIN_LR
    )
    if args.loss == 'l1':
        criterion = torch.nn.L1Loss()
    elif args.loss == 'charbonnier':
        criterion = lambda x, y: charbonnier(x, y, eps=1e-3)

    checkpoint_name = get_checkpoint_name(args.model, config)
    
    # Training loop
    best_psnr = 0.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for lr_img, hr_img in tqdm(train_loader, desc=f'Train {epoch}/{args.epochs}', leave=False):
            lr_img = lr_img.to(DEVICE)
            hr_img = hr_img.to(DEVICE)
            optimizer.zero_grad()
            sr = model(lr_img)
            #print(f"lr_img.shape: {lr_img.shape}, sr.shape: {sr.shape}, hr_img.shape: {hr_img.shape}")
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
            best_epoch = epoch

    # Save checkpoint at end of training
    checkpoint_path = save_checkpoint(
        model, args.model, config, args.checkpoint_dir, checkpoint_name, best_psnr, best_epoch
    )
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")
