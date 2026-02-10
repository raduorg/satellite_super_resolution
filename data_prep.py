#!/usr/bin/env python3
"""
Data preparation script for EuroSAT super resolution.

This script:
1. Scans specified EuroSAT class folders for 64x64 JPG images
2. Creates downscaled 16x16 versions using bicubic interpolation
3. Splits data into train/val/test sets (70%/20%/10%) with stratified sampling
4. Generates CSV files with lr_path, hr_path, class columns
"""

import os
import random
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm


# Default classes to use
DEFAULT_CLASSES = [
    'River',
    'AnnualCrop', 
    'Residential',
    'Highway',
    'Industrial',
    'PermanentCrop'
]

# Image sizes
HR_SIZE = 64  # Original EuroSAT image size
LR_SIZE = 16  # Downscaled size (4x reduction)

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1


def get_image_files(class_dir):
    """Get all JPG image files from a directory."""
    extensions = {'.jpg', '.jpeg', '.png'}
    files = []
    for f in os.listdir(class_dir):
        if Path(f).suffix.lower() in extensions:
            files.append(f)
    return sorted(files)


def create_lr_image(hr_path, lr_path, lr_size=LR_SIZE):
    """Create a low-resolution version of an image using bicubic downscaling."""
    img = Image.open(hr_path).convert('RGB')
    lr_img = img.resize((lr_size, lr_size), Image.BICUBIC)
    lr_img.save(lr_path, 'JPEG', quality=95)


def split_data(files, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=42):
    """
    Split files into train/val/test sets.
    
    Returns:
        Tuple of (train_files, val_files, test_files)
    """
    random.seed(seed)
    files = files.copy()
    random.shuffle(files)
    
    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]
    
    return train_files, val_files, test_files


def prepare_dataset(
    eurosat_dir,
    output_lr_dir,
    output_csv_dir,
    classes=None,
    seed=42
):
    """
    Prepare the EuroSAT dataset for super resolution training.
    
    Args:
        eurosat_dir: Path to EuroSAT directory containing class folders
        output_lr_dir: Path to output directory for LR images
        output_csv_dir: Path to output directory for CSV files
        classes: List of class names to process (default: DEFAULT_CLASSES)
        seed: Random seed for reproducibility
    """
    classes = classes or DEFAULT_CLASSES
    eurosat_dir = Path(eurosat_dir)
    output_lr_dir = Path(output_lr_dir)
    output_csv_dir = Path(output_csv_dir)
    
    # Create output directories
    output_csv_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all data with splits
    train_data = []
    val_data = []
    test_data = []
    
    print(f"Processing {len(classes)} classes from {eurosat_dir}")
    print(f"LR images will be saved to: {output_lr_dir}")
    print(f"CSV files will be saved to: {output_csv_dir}")
    print()
    
    for class_name in classes:
        class_hr_dir = eurosat_dir / class_name
        
        if not class_hr_dir.exists():
            print(f"Warning: Class directory not found: {class_hr_dir}")
            continue
        
        # Create corresponding LR directory
        class_lr_dir = output_lr_dir / class_name
        class_lr_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all images in this class
        image_files = get_image_files(class_hr_dir)
        print(f"  {class_name}: {len(image_files)} images")
        
        # Create LR versions
        for fname in tqdm(image_files, desc=f"  Creating LR images for {class_name}", leave=False):
            hr_path = class_hr_dir / fname
            lr_path = class_lr_dir / fname
            
            if not lr_path.exists():
                create_lr_image(hr_path, lr_path)
        
        # Split into train/val/test
        train_files, val_files, test_files = split_data(image_files, seed=seed)
        
        # Add to datasets with paths relative to project root
        for fname in train_files:
            train_data.append({
                'lr_path': str(output_lr_dir / class_name / fname),
                'hr_path': str(eurosat_dir / class_name / fname),
                'class': class_name
            })
        
        for fname in val_files:
            val_data.append({
                'lr_path': str(output_lr_dir / class_name / fname),
                'hr_path': str(eurosat_dir / class_name / fname),
                'class': class_name
            })
        
        for fname in test_files:
            test_data.append({
                'lr_path': str(output_lr_dir / class_name / fname),
                'hr_path': str(eurosat_dir / class_name / fname),
                'class': class_name
            })
    
    print()
    print(f"Dataset splits:")
    print(f"  Train: {len(train_data)} images")
    print(f"  Val:   {len(val_data)} images")
    print(f"  Test:  {len(test_data)} images")
    
    # Write CSV files
    def write_csv(data, path):
        import csv
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['lr_path', 'hr_path', 'class'])
            writer.writeheader()
            writer.writerows(data)
    
    # Shuffle before writing
    random.seed(seed)
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    write_csv(train_data, output_csv_dir / 'train.csv')
    write_csv(val_data, output_csv_dir / 'val.csv')
    write_csv(test_data, output_csv_dir / 'test.csv')
    
    print()
    print(f"CSV files written:")
    print(f"  {output_csv_dir / 'train.csv'}")
    print(f"  {output_csv_dir / 'val.csv'}")
    print(f"  {output_csv_dir / 'test.csv'}")
    
    return train_data, val_data, test_data


def main():
    parser = argparse.ArgumentParser(
        description='Prepare EuroSAT dataset for super resolution training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--eurosat-dir',
        type=str,
        default='EuroSAT',
        help='Path to EuroSAT directory containing class folders'
    )
    parser.add_argument(
        '--output-lr-dir',
        type=str,
        default='EuroSAT_LR',
        help='Output directory for low-resolution images'
    )
    parser.add_argument(
        '--output-csv-dir',
        type=str,
        default='.',
        help='Output directory for CSV split files'
    )
    parser.add_argument(
        '--classes',
        type=str,
        nargs='+',
        default=DEFAULT_CLASSES,
        help='List of class names to process'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    prepare_dataset(
        eurosat_dir=args.eurosat_dir,
        output_lr_dir=args.output_lr_dir,
        output_csv_dir=args.output_csv_dir,
        classes=args.classes,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
