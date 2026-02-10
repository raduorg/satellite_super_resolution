import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse

from models.srresnet import SRResNetLite
from models.ae import SRAutoEncoderLite
from models.edsr import EDSR
from utils.dataset import SuperResDataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HR_SIZE = 64  # EuroSAT original image size


def geometric_ensemble(model, lr_img, device):
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
    
    if model_type == 'cnn':
        model = SRResNetLite(
            n_feats=arch.get('n_feats', 64),
            n_resblocks=arch.get('n_resblocks', 8),
            res_scale=arch.get('res_scale', 1.0)
        )
    elif model_type == 'edsr':
        model = EDSR(
            n_feats=arch.get('n_feats', 64),
            n_resblocks=arch.get('n_resblocks', 16),
            res_scale=arch.get('res_scale', 1.0)
        )
    elif model_type == 'ae':
        model = SRAutoEncoderLite(
            nf=arch.get('nf', 64)
        )
    else:
        raise ValueError(f"Unknown model type:'{model_type}'. supported models: cnn, edsr, ae")
    
    model = model.to(device)
    
    model.load_state_dict(torch.load(pth_path, map_location=device))
    
    return model, config


def main():
    parser = argparse.ArgumentParser(
        description='Run inference with a trained SR model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        required=True,
        help='Checkpoint name; both .pth and _config.json must exist!!!'
             ''
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='if --checkpoint is just a name'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output_sr',
        help='dir to save super-resolved images'
    )
    parser.add_argument(
        '--tta',
        action='store_true',
        help='enable 8x geometric self-ensemble'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run on validation set instead of test set and output PSNR metrics'
    )
    args = parser.parse_args()
    
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
    
    if args.validate:
        print("VALIDATION MODE")
        
        val_loader = DataLoader(
            SuperResDataset('val.csv'),
            batch_size=1,
            shuffle=False
        )
        
        print(f"\nRunning {'8x TTA ' if args.tta else ''}inference on {len(val_loader)} validation images...")
        psnr_vals = []
        
        with torch.no_grad():
            for lr_img, hr_img in tqdm(val_loader, desc='Validation'):
                hr_img = hr_img.to(DEVICE)
                
                if args.tta:
                    sr = geometric_ensemble(model, lr_img, DEVICE)
                else:
                    sr = model(lr_img.to(DEVICE))
                
                psnr_val = psnr(sr, hr_img).item()
                psnr_vals.append(psnr_val)
        
        mean_psnr = np.mean(psnr_vals)
        std_psnr = np.std(psnr_vals)
        min_psnr = np.min(psnr_vals)
        max_psnr = np.max(psnr_vals)
        
        print("VALIDATION RESULTS")
        print(f"Mean PSNR:  {mean_psnr:.2f} dB")
        print(f"Std PSNR:   {std_psnr:.2f} dB")
        print(f"Min PSNR:   {min_psnr:.2f} dB")
        print(f"Max PSNR:   {max_psnr:.2f} dB")
        
        return 0
    
    test_loader = DataLoader(
        SuperResDataset('test.csv', with_target=False),
        batch_size=1,
        shuffle=False
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    desc = 'Inference (TTA)' if args.tta else 'Inference'
    print(f"\nRunning {'8x TTA ' if args.tta else ''}inference on {len(test_loader)} images...")
    print(f"Output directory: {args.output_dir}")
    
    with torch.no_grad():
        for lr_img, filename in tqdm(test_loader, desc=desc):
            if args.tta:
                sr_tensor = geometric_ensemble(model, lr_img, DEVICE)
                sr = sr_tensor.cpu().squeeze(0).numpy()
            else:
                sr = model(lr_img.to(DEVICE)).cpu().squeeze(0).numpy()
            
            sr_uint8 = (sr * 255.0).round().clip(0, 255).astype(np.uint8)
            # Convert CHW to HWC for PIL
            sr_hwc = sr_uint8.transpose(1, 2, 0)
            
            # Save as JPG with _sr suffix
            output_filename = filename[0].replace('.jpg', '_sr.jpg').replace('.jpeg', '_sr.jpeg')
            output_path = os.path.join(args.output_dir, output_filename)
            Image.fromarray(sr_hwc).save(output_path, 'JPEG', quality=95)
    
    print(f"\nSR images saved to: {args.output_dir}/")
    return 0


if __name__ == '__main__':
    exit(main())
