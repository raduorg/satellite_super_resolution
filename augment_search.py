import os
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

from torchmetrics.image import StructuralSimilarityIndexMeasure

from models.mlp_mixer import MLPMixerSR
from models.edsr import EDSR
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

BEST_CONFIGS = {

    'mlp': [
            {
                'lr': 0.001,
                'batch_size': 64,
                'weight_decay': 0,
                'patch_size': 2,
                'scale': 2,
                'embed_dim': 256,
                'n_layers': 8,
                'token_mlp_dim': 512,
                'channel_mlp_dim': 1024
            },
            {
                'lr': 0.0005,
                'batch_size': 32,
                'weight_decay': 0,
                'patch_size': 2,
                'scale': 2,
                'embed_dim': 128,
                'n_layers': 6,
                'token_mlp_dim': 256,
                'channel_mlp_dim': 512
            },
            {
                'lr': 0.0002,
                'batch_size': 16,
                'weight_decay': 0.0001,
                'patch_size': 2,
                'scale': 2,
                'embed_dim': 128,
                'n_layers': 4,
                'token_mlp_dim': 256,
                'channel_mlp_dim': 512
            },
    ],
    'edsr': [
        {'lr': 5e-4,'batch_size': 32, 'weight_decay': 0.0, 'n_feats': 64, 'n_resblocks': 24, 'res_scale': 0.1},
        {'lr': 5e-4, 'batch_size': 32, 'weight_decay': 0.0,'n_feats': 96, 'n_resblocks': 24,'res_scale': 0.05},
        {'lr': 5e-4, 'batch_size': 32,'weight_decay': 1e-4, 'n_feats': 96,'n_resblocks': 24, 'res_scale': 0.1},
    ],
}

AUGMENTATION_STRATEGIES = {
    'none': {},
    'flip_h': {'horizontal_flip': True},
    'flip_hv': {'horizontal_flip': True, 'vertical_flip': True},
    'flip_rotate': {'horizontal_flip': True, 'vertical_flip': True, 'rotate90': True},
    'full_geometric': {'horizontal_flip': True, 'vertical_flip': True, 'rotate90': True, 'color_jitter': True},
}

def compute_mse(pred, gt):
    return F.mse_loss(pred, gt).item()


def compute_psnr(pred, gt):
    mse = F.mse_loss(pred, gt)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(1.0 / mse)).item()


def evaluate_model(model, val_loader, device):
    model.eval()
    
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
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
            ssim_vals.append(ssim_metric(sr, hr_img).item())
    
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
    if model_type == 'mlp':
            model = MLPMixerSR(
                patch_size=config.get('patch_size', 2),
                embed_dim=config.get('embed_dim', 128),
                n_layers=config.get('n_layers', 6),
                token_mlp_dim=config.get('token_mlp_dim', 256),
                channel_mlp_dim=config.get('channel_mlp_dim', 512),
                p_drop=0.0
            )
    elif model_type == 'edsr':
        model = EDSR(
            n_feats=config.get('n_feats', 64),
            n_resblocks=config.get('n_resblocks', 16),
            res_scale=config.get('res_scale', 1.0)
        )
    else:
        raise ValueError(f"Unknown model type: '{model_type}'. Supported models: mlp, edsr")
    
    return model.to(device)

def get_configs(model_type):
    return BEST_CONFIGS.get(model_type, [])


def run_augmentation_search(model_type, configs, epochs):
    if not configs:
        print(f"No configs provided for {model_type}.skipping.")
        return []
    
    total_runs = len(configs) * len(AUGMENTATION_STRATEGIES)
    
    print(f" configurations to test: {len(configs)}")
    print(f"total runs: {total_runs}")
    print(f"epochs per run: {epochs}")
    print(f"{'='*60}\n")
    
    results = []
    run_count = 0
    
    for config_idx, config in enumerate(configs, 1):
        for aug_name, aug_params in AUGMENTATION_STRATEGIES.items():
            run_count += 1
            print(f"\n[{run_count}/{total_runs}] {model_type.upper()} config {config_idx} + '{aug_name}' augmentation")
            
            train_loader = DataLoader(
                SRDataset('train.csv', augmentation=aug_params),
                batch_size=int(config['batch_size']),
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=True
            )
            val_loader = DataLoader(
                SRDataset('validation.csv', augmentation=None),
                batch_size=int(config['batch_size']),
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=True
            )
            
            model = build_model(model_type, config, DEVICE)
            metrics = train_model(
                model, train_loader, val_loader, config, DEVICE,
                max_epochs=epochs
            )
            result = {
                'model_type': model_type,
                'config_id': config_idx,
                'augmentation': aug_name,
                **config,
                **metrics
            }
            results.append(result)
            print(f"  → MSE: {metrics['mse']:.6f} | PSNR: {metrics['psnr']:.2f} dB | SSIM: {metrics['ssim']:.4f}")
    
    return results


def save_results(results, filename):
    if not results:
        print("No results to save.")
        return None
    df = pd.DataFrame(results)
    df = df.sort_values('psnr', ascending=False)
    df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")

    return df


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--model', choices=['mlp',  'edsr', 'all'],default='all')
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--output-dir', type=str, default='augment_results')    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    models_to_run = ['cnn', 'ae', 'edsr'] if args.model == 'all' else [args.model]
    
    all_results = []
    
    for model_type in models_to_run:
        configs = get_configs(model_type)
        
        if not configs:
            print(f"No configs found for {model_type}, skipping.")
            continue
        
        results = run_augmentation_search( model_type, configs, args.epochs)
        all_results.extend(results)
        
        model_filename = os.path.join(args.output_dir, f'{model_type}_augment_{timestamp}.csv')
        save_results(results, model_filename)
    
    if len(models_to_run) > 1 and all_results:
        combined_filename = os.path.join(args.output_dir, f'combined_augment_{timestamp}.csv')
        save_results(all_results, combined_filename)
        
    print(f"\n{'='*60}")
    print("aug search complete")
    print(f" results saved in:  {args.output_dir}/")


if __name__ == '__main__':
    main()
