import os
import itertools
import random
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse



from models.edsr import EDSR
from models.mlp_mixer import MLPMixerSR
from utils.dataset import SRDataset

SEED = 123
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HR_SIZE = 64   # EuroSAT original image size
LR_SIZE = 16   # Downscaled LR image size

NUM_WORKERS = 4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def compute_mse(pred, gt):
    return F.mse_loss(pred, gt).item()


def compute_psnr(pred, gt):
    mse = F.mse_loss(pred, gt)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(1.0 / mse)).item()


    model.eval()
    def ssim_torch(img1, img2, C1=0.01**2, C2=0.03**2):
        # img1, img2: (N, C, H, W) or (C, H, W), range [0,1]
        if img1.dim() == 4:
            mu1 = img1.mean(dim=[2,3], keepdim=True)
            mu2 = img2.mean(dim=[2,3], keepdim=True)
            sigma1_sq = ((img1 - mu1) ** 2).mean(dim=[2,3], keepdim=True)
            sigma2_sq = ((img2 - mu2) ** 2).mean(dim=[2,3], keepdim=True)
            sigma12 = ((img1 - mu1) * (img2 - mu2)).mean(dim=[2,3], keepdim=True)
            ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                       ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
            return ssim_map.mean().item()
        else:
            mu1 = img1.mean(dim=[1,2], keepdim=True)
            mu2 = img2.mean(dim=[1,2], keepdim=True)
            sigma1_sq = ((img1 - mu1) ** 2).mean(dim=[1,2], keepdim=True)
            sigma2_sq = ((img2 - mu2) ** 2).mean(dim=[1,2], keepdim=True)
            sigma12 = ((img1 - mu1) * (img2 - mu2)).mean(dim=[1,2], keepdim=True)
            ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                       ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
            return ssim_map.mean().item()

    mse_vals = []
    psnr_vals = []
    ssim_vals = []
    with torch.no_grad():
        for lr_img, hr_img in val_loader:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            sr = model(lr_img)
            mse_vals.append(compute_mse(sr, hr_img))
            psnr_vals.append(compute_psnr(sr, hr_img))
            ssim_vals.append(ssim_torch(sr, hr_img))
    return {
        'mse': np.mean(mse_vals),
        'psnr': np.mean(psnr_vals),
        'ssim': np.mean(ssim_vals)
    }


def train_model(model, train_loader, val_loader, config, device, max_epochs):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        betas=(0.9, 0.99),
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-6
    )
    criterion = nn.L1Loss()
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        for lr_img, hr_img in train_loader:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            
            optimizer.zero_grad()
            sr = model(lr_img)
            loss = criterion(sr, hr_img)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
    
    metrics = evaluate_model(model, val_loader, device)
    return metrics


def build_model(model_type, config, device):
    if model_type ==  'edsr':
        model = EDSR(
            n_feats=config.get('n_feats',64),
            n_resblocks=config.get('n_resblocks', 16),
            res_scale=config.get('res_scale',1.0)
        )
    elif model_type == 'mlp':
        model = MLPMixerSR(
            patch_size=config.get('patch_size', 4),
            embed_dim=config.get('embed_dim', 128),
            n_layers=config.get('n_layers', 6),
            token_mlp_dim=config.get('token_mlp_dim', 256),
            channel_mlp_dim=config.get('channel_mlp_dim', 512)
        )
    else:
        raise ValueError(f"Unknown model type: '{model_type}'. Supported models: cnn, edsr, ae")
    
    return model.to(device)


# def get_search_space(model_type):
#     common = {
#         'lr': [1e-3, 2e-4, 1e-4,  5e-4],
#         'batch_size': [16, 32,64],
#         'weight_decay': [0, 1e-4 , 1e-3],
#     }
    
#     if model_type == 'edsr':
#         return {
#             **common,
#             'n_feats': [32, 64,128],
#             'n_resblocks': [8, 16,  32],
#             'res_scale': [0.1,0.5, 1.0],
#         }
#     elif model_type == 'mlp':
#         return {
#             **common,
#             'embed_dim': [128, 256],
#             'n_layers': [4, 6, 8],
#             'token_mlp_dim': [256, 512],
#             'channel_mlp_dim': [512, 1024],
#         }
#     else:
#         raise ValueError(f"Unknown model type: '{model_type}'. Supported models: cnn, edsr, ae")
def get_search_space(model_type: str):
    """ Returns a *list* of concrete configs, not a full Cartesian product. Each config is a dict you can feed directly into argparse.Namespace(**cfg). """
    # ------------------------------------------------------------------
    # common blocks (coupled lr ↔ batch_size)
    # ------------------------------------------------------------------
    #small    = dict(lr=2e-4, batch_size=16)   # conservative
    medium   = dict(lr=5e-4, batch_size=32)   # default
    #large    = dict(lr=1e-3, batch_size=64)   # aggressive
    wd = [0, 1e-4]                            # sensible choices

    if model_type == 'edsr':
        # capacity tiers ------------------------------------------------
        tiny = dict(n_feats=32, n_resblocks=8) # ← legacy (commented)
        base = dict(n_feats=64, n_resblocks=16) # ← legacy (commented)
        deep  = dict(n_feats=64,  n_resblocks=24)     # NEW: deeper, same width
        mid   = dict(n_feats=96,  n_resblocks=24)     # NEW: mid width
        big = dict(n_feats=128, n_resblocks=32) # ← legacy (commented)        ]     
        res_scale_vals = [0.1, 0.05, 0.2]           # only 2 useful values
        capacity_blocks = [
            (deep, medium),
            (mid,  medium),
        ]

        # glue everything together -------------------------------------
        configs = []
        for wd_val in wd:
            for cap, opt in capacity_blocks:
                for rs in res_scale_vals:
                    cfg = dict(weight_decay=wd_val,
                               res_scale=rs,
                               **cap,      # n_feats, n_resblocks
                               **opt)      # lr, batch_size
                    configs.append(cfg)
        return configs

    elif model_type == 'mlp':
            # ------------------------------------------------------------------
            # capacity tiers
            # ------------------------------------------------------------------
            tiny  = dict(embed_dim=128, n_layers=4,
                        token_mlp_dim=256,  channel_mlp_dim=512)
            base  = dict(embed_dim=128, n_layers=6,
                        token_mlp_dim=256,  channel_mlp_dim=512)
            big   = dict(embed_dim=256, n_layers=8,
                        token_mlp_dim=512,  channel_mlp_dim=1024)

            capacity_blocks = [
                (deep,  medium),      # tiny tier + conservative optimiser
                (mid,  medium),     # base tier + medium optimiser
                # (big,   large),      # big tier + aggressive optimiser
            ]

            configs = []

            # ═══════════════════════════════════════════════════════════════
            # 1) OLD CONFIGS (already trained) → kept for reference only
            # The append() lines are commented out, so nothing is added.
            # ═══════════════════════════════════════════════════════════════
            # for wd_val in wd:
            # for cap, opt in capacity_blocks:
            # old_cfg = dict(weight_decay=wd_val,
            # patch_size=4,
            # p_drop=0.0,
            # **cap,
            # **opt)
            # # configs.append(old_cfg) # ← disabled on purpose

            # ═══════════════════════════════════════════════════════════════
            # 2) NEW CONFIGS that we still want to run
            # ═══════════════════════════════════════════════════════════════
            for wd_val in wd:                                 # 2 × weight-decay
                for cap, opt in capacity_blocks:              # 3 × tiers
                    # ---- main variant: patch_size = 2, p_drop = 0 ----------
                    cfg = dict(weight_decay=wd_val,
                            patch_size=2,
                            p_drop=0.0,
                            **cap,
                            **opt)
                    configs.append(cfg)

                    # ---- extra variety: only for the "big" tier -----------
                    if cap is big:
                        extra_cfg = dict(weight_decay=wd_val,
                                        patch_size=4 if wd_val else 2,
                                        p_drop=0.1,
                                        **cap,
                                        **opt)
                        configs.append(extra_cfg)

            return configs
    else:
        raise ValueError('model_type must be "edsr" or "mlp"')


def run_grid_search(model_type, epochs_per_config, max_configs=None):
    all_configs = get_search_space(model_type)
    total_configs = len(all_configs)
    
    if max_configs and len(all_configs) > max_configs:
        all_configs = random.sample(all_configs, max_configs)
    
    print(f"Hyperparameter Search: {model_type}")
    print(f"Search space:")
    # Print a sample of the search space structure
    if all_configs:
        sample_config = all_configs[0]
        for key, values in sample_config.items():
            print(f"  {key}: {values}")
    print(f"Total possible configurations: {total_configs}")
    if max_configs and max_configs < total_configs:
        print(f"Running subset: {len(all_configs)} configs (randomly sampled)")
    else:
        print(f"Running all {len(all_configs)} configurations")
    print(f"Epochs per config: {epochs_per_config}")
    
    results = []
    
    for i, config in enumerate(all_configs, 1):
        print(f"\n[{i}/{len(all_configs)}] Testing config:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        
        train_loader = DataLoader(
            SRDataset('train.csv'),
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        val_loader = DataLoader(
            SRDataset('validation.csv'),
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        
        model = build_model(model_type, config, DEVICE)
        metrics = train_model(
            model, train_loader, val_loader, config, DEVICE,
            max_epochs=epochs_per_config
        )
        
        result = {
            'model_type': model_type,
            **config,
            **metrics
        }
        results.append(result)
        
        print(f"MSE: {metrics['mse']:.6f} | PSNR: {metrics['psnr']:.2f} dB | SSIM: {metrics['ssim']:.4f}")
    
    return results


def save_results(results, filename):
    if not results:
        print("nothing to save")
        return
    
    df = pd.DataFrame(results)
    df = df.sort_values('psnr', ascending=False)
    df.to_csv(filename, index=False)
    print(f"\nresults saved to {filename}")
    
    return df


def print_summary(df, model_type=None):
    if model_type:
        df = df[df['model_type'] == model_type]
        title = f"TOP 5 CONFIGURATIONS - {model_type.upper()}"
    else:
        title = "TOP 5 CONFIGURATIONS - ALL MODELS"
    
    print(title)
    
    display_cols = ['model_type', 'lr', 'batch_size', 'weight_decay', 'mse', 'psnr', 'ssim']
    if model_type in ['cnn', 'edsr']:
        display_cols = ['n_feats', 'n_resblocks', 'res_scale'] + display_cols
    elif model_type == 'ae':
        display_cols = ['nf'] + display_cols
    
    display_cols = [c for c in display_cols if c in df.columns]
    
    print(df.head()[display_cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description='Hyperparameter search for SR models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model', 
        choices=['mlp', 'edsr', 'all'], 
        default='all',
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=30,
    )
    parser.add_argument(
        '--max-configs',
        type=int,
        default=None,
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='hyperparam_results',
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    models_to_run = ['cnn', 'ae', 'edsr'] if args.model == 'all' else [args.model]
    
    all_results = []
    
    for model_type in models_to_run:
        results = run_grid_search(model_type, args.epochs, args.max_configs)
        all_results.extend(results)
        
        model_filename = os.path.join(args.output_dir, f'{model_type}_results_{timestamp}.csv')
        df = save_results(results, model_filename)
        print_summary(df, model_type)
    
    if len(models_to_run) > 1:
        combined_filename = os.path.join(args.output_dir, f'combined_results_{timestamp}.csv')
        df = save_results(all_results, combined_filename)
        print_summary(df)
    
    
    print("hyperparameter search finished")
    print(f"Results saved in: {args.output_dir}/")


if __name__ == '__main__':
    main()
