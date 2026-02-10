import os, numpy as np, pandas as pd, torch
from PIL import Image
import argparse

from infer import load_checkpoint
from utils.dataset import SuperResDataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HR = 64  # EuroSAT original image size
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
args = parser.parse_args()

checkpoint_path = args.checkpoint
if os.path.sep not in checkpoint_path and not checkpoint_path.startswith('.'):
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_path)
if checkpoint_path.endswith('.pth'):
    checkpoint_path = checkpoint_path[:-4]

try:
    model, config = load_checkpoint(checkpoint_path, DEVICE)
except FileNotFoundError as e:
    print(e)
    exit(1)

model.eval()
model_name = config['model_type']
print(f"Model: {model_name.upper()}")
print(f"Architecture: {config['architecture']}")

def load_lr(path):
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img, dtype=np.float32).transpose(2,0,1) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)

# Load first test image from test.csv
test_df = pd.read_csv('test.csv')
first_lr_path = test_df.loc[0, 'lr_path']
lr_tensor = load_lr(first_lr_path).to(DEVICE)

with torch.no_grad():
    sr = model(lr_tensor).cpu().squeeze(0).numpy()

sr_uint8 = (sr * 255).round().clip(0,255).astype(np.uint8)
sr_hwc   = sr_uint8.transpose(1, 2, 0)
flat     = sr_hwc.flatten()

expected_len= HR*HR * 3
if len(flat)==expected_len:
    print(f'flat length correct: {len(flat)}; expected {expected_len})')
else:
    print(f'flat length bad: {len(flat)} (expected {expected_len})')

print(f'first 3 values - RGB of top-left pixel): {flat[:3]}')
print(f'Values 0, {HR*HR}, {2*HR*HR}: {flat[0]}, {flat[HR * HR] if len(flat ) > HR*HR else "N/A"}, {flat[2*HR*HR] if len(flat) > 2*HR*HR else "N/A"}')

recon = flat.reshape(HR, HR, 3)

if np.array_equal(sr_hwc, recon):
    print(' Round-trip lossless.')
else:
    diff = np.abs(sr_hwc.astype(int) - recon.astype(int)).mean()
    print('Round-trip failed: mean absolute diff:', diff)

preview_img = sr_hwc
checkpoint_name = os.path.basename(checkpoint_path)
preview_filename = f"preview_{checkpoint_name}.png"
Image.fromarray(preview_img).save(preview_filename)
print(f'Saved preview: {preview_filename}')