"""
Dataset module for Super-Resolution training and inference.
"""

import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class SRDataset(Dataset):
    """
    Dataset for super resolution that reads LR/HR image pairs from CSV.
    
    CSV format:
        lr_path,hr_path,class (class is optional)
    
    Supports data augmentation applied consistently to both LR and HR images.
    """
    
    def __init__(self, csv_path, root_dir=None, with_target=True, augmentation=None):
        """
        Args:
            csv_path: Path to CSV file with columns 'lr_path', 'hr_path'
            root_dir: Root directory to prepend to paths. If None, paths are used as-is.
            with_target: If True, return (lr, hr) tuple. If False, return (lr, filename).
            augmentation: Dict with augmentation options:
                - horizontal_flip: bool
                - vertical_flip: bool
                - rotate90: bool
                - color_jitter: bool
        """
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.with_target = with_target
        self.augmentation = augmentation or {}

    def __len__(self):
        return len(self.df)

    def _load_image(self, fname, resize=None):
        """Load image, optionally resize, and convert to normalized CHW tensor."""
        if self.root_dir:
            path = os.path.join(self.root_dir, fname)
        else:
            path = fname
        img = Image.open(path).convert('RGB')
        if resize is not None:
            img = img.resize(resize, Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(arr)

    def _apply_augmentation(self, lr, hr):
        """Apply same augmentation to both LR and HR tensors."""
        if self.augmentation.get('horizontal_flip') and random.random() > 0.5:
            lr = torch.flip(lr, dims=[2])
            hr = torch.flip(hr, dims=[2])
        
        if self.augmentation.get('vertical_flip') and random.random() > 0.5:
            lr = torch.flip(lr, dims=[1])
            hr = torch.flip(hr, dims=[1])
        
        if self.augmentation.get('rotate90') and random.random() > 0.5:
            k = random.choice([1, 2, 3])
            lr = torch.rot90(lr, k, dims=[1, 2])
            hr = torch.rot90(hr, k, dims=[1, 2])
        
        if self.augmentation.get('color_jitter'):
            brightness = 1.0 + random.uniform(-0.1, 0.1)
            contrast = 1.0 + random.uniform(-0.1, 0.1)
            lr = torch.clamp((lr - 0.5) * contrast + 0.5 + (brightness - 1.0), 0, 1)
            hr = torch.clamp((hr - 0.5) * contrast + 0.5 + (brightness - 1.0), 0, 1)
        
        return lr, hr

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Always resize LR and HR to expected sizes
        lr = self._load_image(row['lr_path'], resize=(16, 16))
        if self.with_target:
            hr = self._load_image(row['hr_path'], resize=(64, 64))
            if self.augmentation:
                lr, hr = self._apply_augmentation(lr, hr)
            return lr, hr
        return lr, row['lr_path']
