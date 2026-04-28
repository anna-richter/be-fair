import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

# Paths and constants
# start change
# DATA_CSV = "./input/mydataset.csv"  # original
DATA_CSV = "/home/anri21/be-fair/aide/MyData/mydataset.csv"
# end change
# start change
# IMG_DIR = "./input/MyImages"  # original
IMG_DIR = "/home/anri21/be-fair/aide/MyData/MyImages"
# end change
MODEL_PATH = "./working/model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SPLITS = 5
BATCH_SIZE = 64
EPOCHS = 3
LR = 1e-4


# Dataset
class SkinLesionDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(IMG_DIR, f"{row.image_name}.jpg")).convert("RGB")
        label = 1 if row.label == "malignant" else 0
        skin_tone = int(row.skin_tone)
        if self.transform:
            img = self.transform(img)
        return img, label, skin_tone


# Transforms
train_transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
val_transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Load data
df = pd.read_csv(DATA_CSV)
df = df[df.label.isin(["malignant", "benign", "non-neoplastic"])].copy()
y = (df.label == "malignant").astype(int).values


# 4-way TTA flips on tensor: dims (B,C,H,W) flips at dim2/3
def tta_predict_batch(model, x):
    # x: tensor batch
    probs = torch.sigmoid(model(x))
    probs = probs.unsqueeze(0)
    hflip = torch.flip(x, [-1])
    probs = torch.cat([probs, torch.sigmoid(model(hflip)).unsqueeze(0)], 0)
    vflip = torch.flip(x, [-2])
    probs = torch.cat([probs, torch.sigmoid(model(vflip)).unsqueeze(0)], 0)
    hv = torch.flip(x, [-2, -1])
    probs = torch.cat([probs, torch.sigmoid(model(hv)).unsqueeze(0)], 0)
    return probs.mean(0).cpu().numpy()


# Cross‐validation
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
aucs, gaps = [], []

for train_idx, val_idx in skf.split(df, y):
    df_tr, df_va = df.iloc[train_idx], df.iloc[val_idx]
    tr_ds = SkinLesionDataset(df_tr, transform=train_transform)
    va_ds = SkinLesionDataset(df_va, transform=val_transform)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = models.resnext50_32x4d(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    model.train()
    for epoch in range(EPOCHS):
        for imgs, labels, _ in tr_ld:
            imgs, labels = imgs.to(DEVICE), labels.float().to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs).squeeze(1), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Validate with 4-way TTA
    model.eval()
    y_true, y_prob, tones = [], [], []
    with torch.no_grad():
        for imgs, labels, skin in va_ld:
            imgs = imgs.to(DEVICE)
            probs = tta_predict_batch(model, imgs).flatten()
            y_prob.extend(probs.tolist())
            y_true.extend(labels.numpy().tolist())
            tones.extend(skin.tolist())
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    tones = np.array(tones)
    auc = roc_auc_score(y_true, y_prob)
    mask_d = tones <= 3
    mask_l = tones >= 5
    auc_d = (
        roc_auc_score(y_true[mask_d], y_prob[mask_d]) if mask_d.sum() > 0 else np.nan
    )
    auc_l = (
        roc_auc_score(y_true[mask_l], y_prob[mask_l]) if mask_l.sum() > 0 else np.nan
    )
    aucs.append(auc)
    gaps.append(abs(auc_l - auc_d))

mean_auroc = np.nanmean(aucs)
mean_gap = np.nanmean(gaps)
print(f"Mean AUROC: {mean_auroc:.4f}, Mean subgroup AUC gap: {mean_gap:.4f}")

# Retrain on full data
full_ds = SkinLesionDataset(df, transform=train_transform)
full_ld = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
model_full = models.resnext50_32x4d(pretrained=True)
model_full.fc = nn.Linear(model_full.fc.in_features, 1)
model_full = model_full.to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model_full.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
model_full.train()
for epoch in range(EPOCHS):
    for imgs, labels, _ in full_ld:
        imgs, labels = imgs.to(DEVICE), labels.float().to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model_full(imgs).squeeze(1), labels)
        loss.backward()
        optimizer.step()
    scheduler.step()
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save(model_full.state_dict(), MODEL_PATH)


# Prediction function
def predict(folder_path: str) -> dict:
    """
    Args:
        folder_path: path with image files.
    Returns:
        dict of {filename: malignancy_prob}
    """
    mdl = models.resnext50_32x4d(pretrained=False)
    mdl.fc = nn.Linear(mdl.fc.in_features, 1)
    mdl.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    mdl = mdl.to(DEVICE).eval()
    results = {}
    for fname in os.listdir(folder_path):
        if fname.lower().endswith((".jpg", ".png")):
            img = Image.open(os.path.join(folder_path, fname)).convert("RGB")
            x = val_transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                prob = tta_predict_batch(mdl, x).item()
            results[fname] = prob
    return results


# start change
# TEST_DIR = "./input/test_images"  # original
# if os.path.isdir(TEST_DIR):       # original guard
#     preds = predict(TEST_DIR)     # original
#     ...                           # original
_ddi_files = sorted(os.listdir("/home/anri21/be-fair/evaluation/DDI/images"))
_preds = predict("/home/anri21/be-fair/evaluation/DDI/images")
pd.DataFrame({
    "DDI_file": _ddi_files,
    "predicted_probability": [float(_preds[f]) for f in _ddi_files]
}).to_csv("./DDI_predictions.csv", index=False)
# end change
