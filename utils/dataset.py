"""Shared dataset classes for super resolution training and inference."""

import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class SuperResDataset(Dataset):
    """
    Dataset for super resolution that reads LR/HR image pairs from CSV.
    
    CSV format:
        lr_path,hr_path,class (class is optional)
    
    Paths in CSV should be relative to the base_dir or absolute.
    """
    
    def __init__(self, csv_path, base_dir=None, with_target=True):
        """
        Args:
            csv_path: Path to CSV file with columns 'lr_path', 'hr_path' (and optionally 'class')
            base_dir: Base directory to prepend to paths if they are relative. If None, paths are used as-is.
            with_target: If True, return (lr, hr) tuple. If False, return (lr, filename).
        """
        self.df = pd.read_csv(csv_path)
        self.base_dir = base_dir
        self.with_target = with_target

    def __len__(self):
        return len(self.df)

    def _get_path(self, path):
        """Resolve path relative to base_dir if needed."""
        if self.base_dir and not os.path.isabs(path):
            return os.path.join(self.base_dir, path)
        return path

    def _load_image(self, path):
        """Load image and convert to normalized CHW tensor."""
        full_path = self._get_path(path)
        img = Image.open(full_path).convert('RGB')
        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        lr = self._load_image(row['lr_path'])
        
        if self.with_target:
            hr = self._load_image(row['hr_path'])
            return lr, hr
        
        # Return filename for inference/test
        filename = os.path.basename(row['lr_path'])
        return lr, filename


class SuperResDatasetAugmented(Dataset):
    """
    Dataset for super resolution with data augmentation support.
    
    Augmentation is applied consistently to both LR and HR images.
    """
    
    # Default augmentation parameters
    COLOR_JITTER_BRIGHTNESS = 0.1
    COLOR_JITTER_CONTRAST = 0.1
    
    def __init__(self, csv_path, base_dir=None, augmentation=None, with_target=True):
        """
        Args:
            csv_path: Path to CSV file with columns 'lr_path', 'hr_path'
            base_dir: Base directory to prepend to paths if they are relative
            augmentation: Dict with augmentation options:
                - horizontal_flip: bool
                - vertical_flip: bool
                - rotate90: bool
                - color_jitter: bool
            with_target: If True, return (lr, hr) tuple
        """
        self.df = pd.read_csv(csv_path)
        self.base_dir = base_dir
        self.augmentation = augmentation or {}
        self.with_target = with_target

    def __len__(self):
        return len(self.df)

    def _get_path(self, path):
        """Resolve path relative to base_dir if needed."""
        if self.base_dir and not os.path.isabs(path):
            return os.path.join(self.base_dir, path)
        return path

    def _load_image(self, path):
        """Load image as PIL Image."""
        full_path = self._get_path(path)
        return Image.open(full_path).convert('RGB')

    def _apply_augmentation(self, lr_img, hr_img):
        """Apply augmentation consistently to both LR and HR images."""
        if self.augmentation.get('horizontal_flip', False):
            if random.random() > 0.5:
                lr_img = TF.hflip(lr_img)
                hr_img = TF.hflip(hr_img)
        
        if self.augmentation.get('vertical_flip', False):
            if random.random() > 0.5:
                lr_img = TF.vflip(lr_img)
                hr_img = TF.vflip(hr_img)
        
        if self.augmentation.get('rotate90', False):
            k = random.randint(0, 3)
            if k > 0:
                lr_img = TF.rotate(lr_img, angle=90 * k)
                hr_img = TF.rotate(hr_img, angle=90 * k)
        
        if self.augmentation.get('color_jitter', False):
            brightness_factor = 1.0 + random.uniform(
                -self.COLOR_JITTER_BRIGHTNESS, 
                self.COLOR_JITTER_BRIGHTNESS
            )
            contrast_factor = 1.0 + random.uniform(
                -self.COLOR_JITTER_CONTRAST, 
                self.COLOR_JITTER_CONTRAST
            )
            
            lr_img = TF.adjust_brightness(lr_img, brightness_factor)
            lr_img = TF.adjust_contrast(lr_img, contrast_factor)
            hr_img = TF.adjust_brightness(hr_img, brightness_factor)
            hr_img = TF.adjust_contrast(hr_img, contrast_factor)
        
        return lr_img, hr_img

    def _to_tensor(self, img):
        """Convert PIL Image to normalized CHW tensor."""
        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        lr_img = self._load_image(row['lr_path'])
        
        if self.with_target:
            hr_img = self._load_image(row['hr_path'])
            
            if self.augmentation:
                lr_img, hr_img = self._apply_augmentation(lr_img, hr_img)
            
            return self._to_tensor(lr_img), self._to_tensor(hr_img)
        
        filename = os.path.basename(row['lr_path'])
        return self._to_tensor(lr_img), filename
