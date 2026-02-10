import os
import itertools
import random
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse

from torchmetrics.image import StructuralSimilarityIndexMeasure

from models.srresnet import SRResNetLite
from models.ae import SRAutoEncoderLite
from models.edsr import EDSR

SEED = 123
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HR_SIZE = 128
LR_SIZE = 32

NUM_WORKERS = 4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

class SRDataset(Dataset):
    def __init__(self, csv_path, root_dir, with_target=True):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.with_target = with_target

    def __len__(self):
        return len(self.df)

    def _load_image(self, fname):
        img = Image.open(os.path.join(self.root_dir, fname)).convert('RGB')
        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        lr = self._load_image(row['input_image'])
        if self.with_target:
            hr = self._load_image(row['target_image'])
            return lr, hr
        return lr, row['id']


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
    if model_type == 'cnn':
        model = SRResNetLite(
            n_feats=config.get( 'n_feats', 64),
            n_resblocks=config.get('n_resblocks',8),
            res_scale=config.get('res_scale', 1.0)
        )
    elif model_type == 'edsr':
        model = EDSR(
            n_feats=config.get('n_feats',64),
            n_resblocks=config.get('n_resblocks', 16),
            res_scale=config.get('res_scale',1.0)
        )
    elif model_type == 'ae':
        model = SRAutoEncoderLite(
            nf=config.get('nf', 64 )
        )
    else:
        raise ValueError(f"Unknown model type: '{model_type}'. Supported models: cnn, edsr, ae")
    
    return model.to(device)


def get_search_space(model_type):
    common = {
        'lr': [1e-3, 2e-4, 1e-4,  5e-4],
        'batch_size': [16, 32,64],
        'weight_decay': [0, 1e-4 , 1e-3],
    }
    
    if model_type == 'cnn':
        return {
            **common,
            'n_feats': [32, 64, 128],
            'n_resblocks': [4, 8,12],
            'res_scale': [0.1,0.5,1.0],
        }
    elif model_type == 'edsr':
        return {
            **common,
            'n_feats': [32, 64,128],
            'n_resblocks': [8, 16,  32],
            'res_scale': [0.1,0.5, 1.0],
        }
    elif model_type == 'ae':
        return {
            **common,
            'nf': [32, 64, 128],
        }
    else:
        raise ValueError(f"Unknown model type: '{model_type}'. Supported models: cnn, edsr, ae")


def generate_configs(search_space):
    keys = list(search_space.keys())
    values = list(search_space.values())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def run_grid_search(model_type, epochs_per_config, max_configs=None):
    search_space = get_search_space(model_type)
    all_configs = generate_configs(search_space)
    total_configs = len(all_configs)
    
    if max_configs and len(all_configs) > max_configs:
        all_configs = random.sample(all_configs, max_configs)
    
    print(f"Hyperparameter Search: {model_type}")
    print(f"Search space:")
    for key, values in search_space.items():
        print(f"  {key}: {values}")
    print(f"Total possible configurations: {total_configs}")
    if max_configs and max_configs < total_configs:
        print(f"Running subset: {len(all_configs)} configs (rzndomly sampled)")
    else:
        print(f"Running all {len(all_configs)} configurations")
    print(f"Epochs per config: {epochs_per_config}")
    
    results = []
    
    for i, config in enumerate(all_configs, 1):
        print(f"\n[{i}/{len(all_configs)}] Testing config:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        
        train_loader = DataLoader(
            SRDataset('train.csv', 'train'),
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        val_loader = DataLoader(
            SRDataset('validation.csv', 'validation'),
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
        choices=['cnn', 'ae', 'edsr', 'all'], 
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
