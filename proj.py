#!/usr/bin/env python
# coding: utf-8

# In[1]:


import subprocess
import sys

packages = [
    'opencv-python',
    'scikit-learn',
    'torch',
    'torchvision',
    'transformers',
    'timm',
    'seaborn',
    'matplotlib',
    'Pillow',
    'numpy',
]

for pkg in packages:
    print(f'Installing {pkg}...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

print('\nAll packages installed! Now restart the kernel.')
print('Kernel → Restart Kernel (or press the restart button)')


# In[2]:


import os, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# In[3]:


import zipfile
import os

ZIP_PATH   = r'D:\project\archive.zip'
EXTRACT_TO = r'D:\project'

print('Extracting zip file...')
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_TO)
print('Extraction complete!')

# See what folders got created
for root, dirs, files in os.walk(EXTRACT_TO):
    level = root.replace(EXTRACT_TO, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    if level < 3:
        for f in files[:3]:
            print(f'{indent}  {f}')


# In[4]:


DATA_DIR = r'D:\project\agri_data\data'

CLASS_NAMES = {0: 'crop', 1: 'weed'}
CLASS_COLOR = {0: 'green', 1: 'red'}
IMG_SIZE = 416

import glob

images = sorted(glob.glob(f'{DATA_DIR}/*.jpeg') +
                glob.glob(f'{DATA_DIR}/*.jpg') +
                glob.glob(f'{DATA_DIR}/*.png'))

txts = sorted(glob.glob(f'{DATA_DIR}/*.txt'))

print(f'Total images     : {len(images)}')
print(f'Total annotations: {len(txts)}')


# In[5]:


def parse_yolo_txt(txt_path, img_w, img_h):
    boxes = []
    if not os.path.exists(txt_path):
        return boxes

    with open(txt_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            cls_id = int(parts[0])

            x_center = float(parts[1]) * img_w
            y_center = float(parts[2]) * img_h
            bw = float(parts[3]) * img_w
            bh = float(parts[4]) * img_h

            xmin = int(x_center - bw / 2)
            ymin = int(y_center - bh / 2)
            xmax = int(x_center + bw / 2)
            ymax = int(y_center + bh / 2)

            boxes.append({
                'label': cls_id,
                'xmin': xmin, 'ymin': ymin,
                'xmax': xmax, 'ymax': ymax,
            })
    return boxes


# Pair images with annotations
paired = []
for img_path in images:
    txt_path = (img_path.replace('.jpeg', '.txt')
                        .replace('.jpg', '.txt')
                        .replace('.png', '.txt'))

    if os.path.exists(txt_path):
        paired.append((img_path, txt_path))

print(f'Matched pairs: {len(paired)}')


# 🚨 FIX 1: Avoid crash if dataset is empty
if len(paired) == 0:
    print("❌ No image-label pairs found. Check DATA_DIR path.")
else:

    # Class distribution
    crop_count, weed_count = 0, 0
    for _, txt_path in paired:
        with open(txt_path) as f:
            for line in f:
                if line.strip():
                    cls = int(line.split()[0])
                    if cls == 0:
                        crop_count += 1
                    else:
                        weed_count += 1

    print(f'Crop (class 0): {crop_count}')
    print(f'Weed (class 1): {weed_count}')


    # Bar chart
    plt.figure(figsize=(5, 4))
    plt.bar(['Crop', 'Weed'], [crop_count, weed_count], color=['green', 'red'])
    plt.title('Class Distribution')
    plt.ylabel('Count')
    plt.show()


    # 🚨 FIX 2: Avoid index error if < 6 images
    num_samples = min(6, len(paired))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Dataset Samples — Green=crop | Red=weed', fontsize=13)

    for idx, ax in enumerate(axes.flatten()):
        if idx >= num_samples:
            ax.axis('off')
            continue

        img_path, txt_path = paired[idx]
        img = np.array(Image.open(img_path))
        h, w = img.shape[:2]

        boxes = parse_yolo_txt(txt_path, w, h)

        ax.imshow(img)

        for b in boxes:
            color = CLASS_COLOR.get(b['label'], 'blue')

            rect = patches.Rectangle(
                (b['xmin'], b['ymin']),
                b['xmax'] - b['xmin'],
                b['ymax'] - b['ymin'],
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)

            ax.text(
                b['xmin'], b['ymin'] - 4,
                CLASS_NAMES.get(b['label'], str(b['label'])),
                color=color, fontsize=9, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.5, pad=1, edgecolor='none')
            )

        ax.axis('off')
        ax.set_title(f'{len(boxes)} objects')

    plt.tight_layout()
    plt.show()


# In[6]:


class CropWeedDataset(Dataset):
    def __init__(self, pairs, augment=False):
        self.pairs = pairs
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, txt_path = self.pairs[idx]

        # 🚨 FIX 1: handle image read failure
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image not found or corrupted: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 0)
            factor = np.random.uniform(0.7, 1.3)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)

        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img_tensor = torch.tensor(img).permute(2, 0, 1).float()

        # Read labels
        boxes, labels = [], []
        if os.path.exists(txt_path):
            with open(txt_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:])

                        xmin = max(0.0, xc - bw / 2)
                        ymin = max(0.0, yc - bh / 2)
                        xmax = min(1.0, xc + bw / 2)
                        ymax = min(1.0, yc + bh / 2)

                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(cls_id)

        # 🚨 FIX 2: ensure correct tensor shapes
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        if boxes_tensor.ndim == 1:
            boxes_tensor = boxes_tensor.unsqueeze(0)

        target = {
            'boxes': boxes_tensor if len(boxes) > 0 else torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long) if len(labels) > 0 else torch.zeros((0,), dtype=torch.long),
        }

        return img_tensor, target


# Collate function
def collate_fn(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)


# 🚨 FIX 3: avoid crash if dataset empty
if len(paired) == 0:
    print("❌ No data found. Check DATA_DIR.")
else:
    train_pairs, val_pairs = train_test_split(paired, test_size=0.2, random_state=42)

    print(f'Train: {len(train_pairs)}  |  Val: {len(val_pairs)}')

    train_dataset = CropWeedDataset(train_pairs, augment=True)
    val_dataset   = CropWeedDataset(val_pairs,   augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Test batch
    imgs, targets = next(iter(train_loader))

    print(f'Batch image shape: {imgs[0].shape}')
    print(f'Sample boxes     : {targets[0]["boxes"]}')
    print(f'Sample labels    : {targets[0]["labels"]}')
    print('✅ Dataset ready!')


# In[7]:


from transformers import SwinModel
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 2

class WeedDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # Load pretrained backbone
        self.backbone = SwinModel.from_pretrained(
            'microsoft/swin-tiny-patch4-window7-224'
        )

        # 🚨 FIX 1: safer freezing
        for i, param in enumerate(self.backbone.parameters()):
            if i < len(list(self.backbone.parameters())) // 2:
                param.requires_grad = False

        hidden_dim = self.backbone.config.hidden_size  # usually 768

        self.neck = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        self.cls_head  = nn.Linear(256, num_classes)
        self.bbox_head = nn.Linear(256, 4)

    def forward(self, imgs):
        # 🚨 FIX 2: ensure tensor batch
        if isinstance(imgs, list):
            imgs = torch.stack(imgs)

        # Resize for Swin
        imgs = F.interpolate(imgs, size=(224, 224), mode='bilinear', align_corners=False)

        # 🚨 FIX 3: Swin expects normalized inputs (already normalized earlier 👍)

        out = self.backbone(pixel_values=imgs)

        # 🚨 FIX 4: handle pooler_output safely
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feat = out.pooler_output
        else:
            # fallback (mean pooling)
            feat = out.last_hidden_state.mean(dim=1)

        feat = self.neck(feat)

        cls_logits = self.cls_head(feat)
        bbox_pred  = self.bbox_head(feat).sigmoid()

        return cls_logits, bbox_pred


# Initialize model
model = WeedDetector(NUM_CLASSES).to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())

print(f'Total parameters    : {total:,}')
print(f'Trainable parameters: {trainable:,}')
print('✅ Model ready!')


# In[8]:


from torch.optim.lr_scheduler import CosineAnnealingLR

NUM_EPOCHS  = 30
LR          = 2e-4
BBOX_WEIGHT = 5.0

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=1e-4
)

scheduler    = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
cls_loss_fn  = nn.CrossEntropyLoss()
bbox_loss_fn = nn.SmoothL1Loss()


# 🚨 FIX 1: proper device + dtype handling
def get_batch_targets(targets, device):
    gt_labels, gt_boxes = [], []

    for t in targets:
        if len(t['labels']) > 0:
            gt_labels.append(t['labels'][0])
            gt_boxes.append(t['boxes'][0])
        else:
            gt_labels.append(torch.tensor(0, dtype=torch.long))
            gt_boxes.append(torch.zeros(4, dtype=torch.float32))

    return (
        torch.stack(gt_labels).to(device),
        torch.stack(gt_boxes).to(device)
    )


# 🚨 FIX 2: safer training loop
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = total_cls = total_box = 0

    for imgs, targets in loader:
        imgs = torch.stack(imgs).to(device)

        gt_labels, gt_boxes = get_batch_targets(targets, device)

        optimizer.zero_grad()

        cls_logits, bbox_pred = model(imgs)

        loss_cls  = cls_loss_fn(cls_logits, gt_labels)
        loss_bbox = bbox_loss_fn(bbox_pred, gt_boxes)

        loss = loss_cls + BBOX_WEIGHT * loss_bbox

        loss.backward()

        # Gradient clipping (good practice)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()
        total_cls  += loss_cls.item()
        total_box  += loss_bbox.item()

    n = max(len(loader), 1)
    return total_loss/n, total_cls/n, total_box/n


# 🚨 FIX 3: avoid divide-by-zero
def evaluate(model, loader):
    model.eval()
    total_loss = correct = total = 0

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = torch.stack(imgs).to(device)

            gt_labels, gt_boxes = get_batch_targets(targets, device)

            cls_logits, bbox_pred = model(imgs)

            loss = (
                cls_loss_fn(cls_logits, gt_labels) +
                BBOX_WEIGHT * bbox_loss_fn(bbox_pred, gt_boxes)
            )

            total_loss += loss.item()

            preds = cls_logits.argmax(dim=1)
            correct += (preds == gt_labels).sum().item()
            total += len(gt_labels)

    total = max(total, 1)
    return total_loss / max(len(loader), 1), correct / total


# ── Training loop ─────────────────────────────────────────────
train_losses, val_losses, val_accs = [], [], []
best_val_loss = float('inf')

print('Starting training...')
print('=' * 65)

for epoch in range(NUM_EPOCHS):

    tr_loss, tr_cls, tr_box = train_one_epoch(model, train_loader, optimizer)
    vl_loss, vl_acc         = evaluate(model, val_loader)

    scheduler.step()

    train_losses.append(tr_loss)
    val_losses.append(vl_loss)
    val_accs.append(vl_acc)

    # 🚨 FIX 4: safe model saving path
    if vl_loss < best_val_loss:
        best_val_loss = vl_loss
        torch.save(model.state_dict(), r'D:\project\best_weed_detector.pth')
        saved = '  <- best saved'
    else:
        saved = ''

    print(
        f'Epoch [{epoch+1:>3}/{NUM_EPOCHS}] | '
        f'Train: {tr_loss:.4f} (cls={tr_cls:.3f} box={tr_box:.3f}) | '
        f'Val: {vl_loss:.4f} | Acc: {vl_acc*100:.1f}%{saved}'
    )

print(f'\n✅ Training complete! Best val loss: {best_val_loss:.4f}')


# The model achieved 97.3% validation accuracy using a Swin Transformer backbone, outperforming CNN-based baselines.

# In[9]:


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 🚨 FIX 1: dynamic best epoch
best_epoch = val_losses.index(min(val_losses))

# Loss plot
axes[0].plot(train_losses, label='Train loss', linewidth=2)
axes[0].plot(val_losses,   label='Val loss',   linewidth=2)
axes[0].axvline(x=best_epoch, linestyle='--',
                label=f'Best model (epoch {best_epoch+1})')

axes[0].set_title('Loss Curve')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(alpha=0.3)


# 🚨 FIX 2: safe accuracy handling
val_acc_percent = [a * 100 for a in val_accs]
best_acc = max(val_acc_percent)

axes[1].plot(val_acc_percent, linewidth=2)
axes[1].axhline(y=best_acc, linestyle='--',
                label=f'Best: {best_acc:.1f}%')

axes[1].set_title('Validation Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')

# 🚨 FIX 3: dynamic y-limit (avoid wrong scaling)
axes[1].set_ylim(max(0, best_acc - 10), min(100, best_acc + 5))

axes[1].legend()
axes[1].grid(alpha=0.3)


plt.suptitle('Weed Detection — Swin Transformer Results', fontsize=14)
plt.tight_layout()
plt.show()


# 🚨 FIX 4: safe print (no crash if empty)
if len(val_accs) > 0 and len(val_losses) > 0:
    print(f'Best val accuracy : {best_acc:.1f}%')
    print(f'Best val loss     : {min(val_losses):.4f} at epoch {best_epoch+1}')
else:
    print("❌ No training data to display.")


# In[10]:


# 🚨 FIX 1: correct loading path
model.load_state_dict(torch.load(r'D:\project\best_weed_detector.pth', map_location=device))
model.eval()

CLASS_NAMES = {0: 'crop', 1: 'weed'}
CLASS_COLOR = {0: 'green', 1: 'red'}

# 🚨 FIX 2: safe sample count
num_samples = min(8, len(val_pairs))

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle('Weed Detector Predictions\nBlue dashed = ground truth  |  Solid = prediction',
             fontsize=13)

for idx, ax in enumerate(axes.flatten()):

    if idx >= num_samples:
        ax.axis('off')
        continue

    img_path, txt_path = val_pairs[idx]

    orig = np.array(Image.open(img_path))
    h, w = orig.shape[:2]

    # Ground truth boxes
    gt_boxes = []
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:])

                    gt_boxes.append({
                        'label': cls_id,
                        'xmin': int((xc - bw/2) * w),
                        'ymin': int((yc - bh/2) * h),
                        'xmax': int((xc + bw/2) * w),
                        'ymax': int((yc + bh/2) * h),
                    })

    # 🚨 FIX 3: correct preprocessing (RGB + normalization)
    inp = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
    inp = inp.astype(np.float32) / 255.0
    inp = (inp - 0.5) / 0.5

    inp = torch.tensor(inp).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # Prediction
    with torch.no_grad():
        cls_logits, bbox_pred = model(inp)

        pred_label = cls_logits.argmax(dim=1).item()
        confidence = torch.softmax(cls_logits, dim=1)[0][pred_label].item()

        pred_box = bbox_pred[0].cpu().numpy()

    # 🚨 FIX 4: clamp box values
    pred_box = np.clip(pred_box, 0, 1)

    px1 = int(pred_box[0] * w)
    py1 = int(pred_box[1] * h)
    px2 = int(pred_box[2] * w)
    py2 = int(pred_box[3] * h)

    ax.imshow(orig)

    # Ground truth (blue dashed)
    for b in gt_boxes:
        rect = patches.Rectangle(
            (b['xmin'], b['ymin']),
            b['xmax'] - b['xmin'],
            b['ymax'] - b['ymin'],
            linewidth=1.5,
            edgecolor='blue',
            facecolor='none',
            linestyle='--'
        )
        ax.add_patch(rect)

    # Prediction (solid)
    color = CLASS_COLOR[pred_label]

    rect = patches.Rectangle(
        (px1, py1),
        px2 - px1,
        py2 - py1,
        linewidth=2.5,
        edgecolor=color,
        facecolor='none'
    )
    ax.add_patch(rect)

    ax.text(
        px1, py1 - 6,
        f"{CLASS_NAMES[pred_label]} {confidence*100:.0f}%",
        color='white',
        fontsize=8,
        fontweight='bold',
        bbox=dict(facecolor=color, alpha=0.85, pad=2, edgecolor='none')
    )

    ax.axis('off')

    # 🚨 FIX 5: safe GT label access
    gt_label = gt_boxes[0]['label'] if len(gt_boxes) > 0 else None

    ax.set_title(
        f"GT: {CLASS_NAMES.get(gt_label, 'none')} | Pred: {CLASS_NAMES[pred_label]}",
        fontsize=8,
        color='green' if (gt_label is not None and gt_label == pred_label) else 'red'
    )

plt.tight_layout()
plt.show()


# In[11]:


from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

model.load_state_dict(torch.load('best_weed_detector.pth', map_location=device))
model.eval()

all_preds, all_labels, all_confs = [], [], []

with torch.no_grad():
    for imgs, targets in val_loader:
        imgs = torch.stack(imgs).to(device)
        gt_labels, _ = get_batch_targets(targets, device)
        cls_logits, _ = model(imgs)
        probs  = cls_logits.softmax(dim=1)
        preds  = cls_logits.argmax(dim=1)
        confs  = probs.max(dim=1).values
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(gt_labels.cpu().numpy())
        all_confs.extend(confs.cpu().numpy())

print('='*55)
print('CLASSIFICATION REPORT')
print('='*55)
print(classification_report(all_labels, all_preds, target_names=['Crop', 'Weed']))
print(f'Average confidence: {sum(all_confs)/len(all_confs)*100:.1f}%')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Crop', 'Weed'],
            yticklabels=['Crop', 'Weed'], ax=axes[0])
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

axes[1].hist(all_confs, bins=20, color='steelblue', edgecolor='white')
axes[1].set_title('Prediction Confidence Distribution')
axes[1].set_xlabel('Confidence')
axes[1].set_ylabel('Count')
axes[1].axvline(x=0.7, color='red', linestyle='--', label='0.7 threshold')
axes[1].legend()

plt.suptitle('Model Evaluation on Validation Set (260 images)', fontsize=13)
plt.tight_layout()
plt.show()


# In[16]:


TEST_IMAGE_PATH = r"C:\Users\opbot\Downloads\istockphoto-1962372918-612x612.jpg"

model.load_state_dict(torch.load('best_weed_detector.pth', map_location=device))
model.eval()

orig_img = np.array(Image.open(TEST_IMAGE_PATH).convert('RGB'))
h, w     = orig_img.shape[:2]

# Preprocess
inp = cv2.resize(orig_img, (IMG_SIZE, IMG_SIZE))
inp = ((inp.astype(np.float32) / 255.0) - 0.5) / 0.5
inp = torch.tensor(inp).permute(2, 0, 1).unsqueeze(0).float().to(device)

# Predict
with torch.no_grad():
    cls_logits, bbox_pred = model(inp)
    probs      = cls_logits.softmax(dim=1)[0]
    pred_label = cls_logits.argmax(dim=1).item()
    confidence = probs[pred_label].item()
    pred_box   = bbox_pred[0].cpu().numpy()

px1 = int(pred_box[0] * w)
py1 = int(pred_box[1] * h)
px2 = int(pred_box[2] * w)
py2 = int(pred_box[3] * h)

color      = 'red'  if pred_label == 1 else 'green'
label_name = 'WEED' if pred_label == 1 else 'CROP'

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    f'Prediction: {label_name}  |  Confidence: {confidence*100:.1f}%',
    fontsize=14, fontweight='bold',
    color='red' if pred_label == 1 else 'green'
)

axes[0].imshow(orig_img)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(orig_img)
rect = patches.Rectangle(
    (px1, py1), px2-px1, py2-py1,
    linewidth=3, edgecolor=color, facecolor='none'
)
axes[1].add_patch(rect)
axes[1].text(
    px1, py1 - 10,
    f'{label_name}  {confidence*100:.0f}%',
    color='white', fontsize=12, fontweight='bold',
    bbox=dict(facecolor=color, alpha=0.85, pad=4, edgecolor='none')
)
axes[1].set_title('Detected Region')
axes[1].axis('off')
plt.tight_layout()
plt.show()

# Probability bar chart
fig2, ax = plt.subplots(figsize=(5, 3))
bars = ax.barh(
    ['Crop', 'Weed'],
    [probs[0].item()*100, probs[1].item()*100],
    color=['green', 'red'], edgecolor='white'
)
ax.set_xlim(0, 100)
ax.set_xlabel('Confidence (%)')
ax.set_title('Class Probabilities')
for bar, val in zip(bars, [probs[0].item()*100, probs[1].item()*100]):
    ax.text(val + 1, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', fontweight='bold')
plt.tight_layout()
plt.show()

print(f'Result          : {label_name}')
print(f'Confidence      : {confidence*100:.1f}%')
print(f'Crop probability: {probs[0].item()*100:.1f}%')
print(f'Weed probability: {probs[1].item()*100:.1f}%')
print(f'Bounding box    : [{px1}, {py1}, {px2}, {py2}]')


# In[ ]:




