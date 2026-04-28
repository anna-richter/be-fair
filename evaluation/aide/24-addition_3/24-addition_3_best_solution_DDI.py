import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models

# Paths and parameters
# start change
# DATA_CSV = "./input/mydataset.csv"  # original
DATA_CSV = "/home/anri21/be-fair/aide/MyData/mydataset.csv"
# end change
# start change
# IMG_DIR = "./input/MyImages"  # original
IMG_DIR = "/home/anri21/be-fair/aide/MyData/MyImages"
# end change
# NOTE: This script is INCOMPLETE — no full-data retrain, no model save, no predict() function.
# Only training path changes have been applied. Manual completion required.
TEST_DIR = "./input/test"
MODEL_PATH = "./working/efficientnet_b3_randaug.pth"
SUBMISSION_PATH = "./working/submission.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FOLDS = 5
BATCH_SIZE = 16
EPOCHS_CV = 3
EPOCHS_FULL = 5
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Load metadata
df = pd.read_csv(DATA_CSV)
df["target"] = (df["label"] == "malignant").astype(int)
tone_counts = df["skin_tone"].value_counts().to_dict()
tone_weights = {tone: 1.0 / count for tone, count in tone_counts.items()}
df["sample_weight"] = df["skin_tone"].map(lambda x: tone_weights.get(x, 1.0))


# Dataset
class SkinDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(
            os.path.join(self.img_dir, row["image_name"] + ".jpg")
        ).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(row["target"], dtype=torch.float32)
        return img, label


# Transforms with RandAugment
train_tf = transforms.Compose(
    [
        transforms.Resize((300, 300)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
val_tf = transforms.Compose(
    [
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction="mean"):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean() if self.reduction == "mean" else loss.sum()


# Multi-TTA evaluation
def eval_model_multitta(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            views = [
                imgs,
                torch.flip(imgs, [3]),
                torch.rot90(imgs, 1, [2, 3]),
                torch.rot90(imgs, 2, [2, 3]),
                torch.rot90(imgs, 3, [2, 3]),
            ]
            logits_sum = sum(model(v).squeeze(1) for v in views)
            probs = torch.sigmoid(logits_sum / len(views))
            preds.extend(probs.cpu().tolist())
            trues.extend(labels.tolist())
    return roc_auc_score(trues, preds)


# Training loop
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


# 5-fold CV
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
aucs = []
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["target"])):
    df_train, df_val = df.iloc[train_idx], df.iloc[val_idx]
    train_ds = SkinDataset(df_train, IMG_DIR, train_tf)
    val_ds = SkinDataset(df_val, IMG_DIR, val_tf)
    sampler = WeightedRandomSampler(
        df_train["sample_weight"].values, len(df_train), True
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = models.efficientnet_b3(pretrained=True)
    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, 1)
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = FocalLoss(alpha=0.25, gamma=2)
    for _ in range(EPOCHS_CV):
        train_one_epoch(model, train_loader, optimizer, criterion)
    auc = eval_model_multitta(model, val_loader)
    print(f"Fold {fold+1} AUROC with RandAugment: {auc:.4f}")
    aucs.append(auc)

mean_auc, std_auc = np.mean(aucs), np.std(aucs)
print(f"Mean CV AUROC with RandAugment: {mean_auc:.4f} ± {std_auc:.4f}")
