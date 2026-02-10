
import os
import csv
import random
from datetime import datetime
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
from models.srresnet import SRResNetLite
from models.ae import SRAutoEncoderLite

SEED        = 123
BATCH_SIZE  = 32
LR          = 2e-4
BASE_LR = 2e-4          # start LR
MIN_LR  = 1e-6          # final LR at the end of training
WEIGHT_DECAY = 1e-4

EPOCHS      = 100
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
HR_SIZE     = 128
LR_SIZE     = 32
NUM_WORKERS = 4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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


train_loader = DataLoader(
    SRDataset('train.csv', 'train'),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)
val_loader = DataLoader(
    SRDataset('validation.csv', 'validation'),
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
parser.add_argument('--model', choices=['cnn', 'ae'], default='cnn', help='Model to use: cnn or ae')
args = parser.parse_args()

# Model selection
if args.model == 'cnn':
    model = SRResNetLite().to(DEVICE)
    model_name = 'cnn'
elif args.model == 'ae':
    model = SRAutoEncoderLite().to(DEVICE)
    model_name = 'ae'
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
    SRDataset('test_input.csv', 'test_input', with_target=False),
    batch_size=1,
    shuffle=False
)

# Handle existing submission.csv
submissions_dir = 'submissions'
if not os.path.exists(submissions_dir):
    os.makedirs(submissions_dir)

if os.path.exists('submission.csv'):
    # Get file modification time
    file_stat = os.stat('submission.csv')
    creation_time = datetime.fromtimestamp(file_stat.st_mtime)
    timestamp = creation_time.strftime('%m%d%H%M')
    
    # Move to submissions directory with timestamp
    old_name = f'submission_{timestamp}.csv'
    old_path = os.path.join(submissions_dir, old_name)
    shutil.move('submission.csv', old_path)
    print(f'Moved existing submission to {old_path}')

submission_rows = []
total_pixels = HR_SIZE * HR_SIZE * 3
with torch.no_grad():
    for lr_img, img_id in tqdm(test_loader, desc='Inference'):
        sr = model(lr_img.to(DEVICE)).cpu().squeeze(0).numpy()
        sr_uint8 = (sr * 255.0).round().clip(0, 255).astype(np.uint8)
        submission_rows.append([img_id.item()] + sr_uint8.flatten().tolist())

with open('submission.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id'] + [f'pixel_{i}' for i in range(total_pixels)])
    writer.writerows(submission_rows)

print('Finished. Submission file created.')