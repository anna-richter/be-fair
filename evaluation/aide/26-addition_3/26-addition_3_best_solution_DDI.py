import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import timm


class FocalLoss(nn.Module):
    def __init__(self, alpha=(0.25, 0.75), gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = torch.exp(-bce)
        focal_term = (1 - p_t) ** self.gamma
        loss = focal_term * bce
        alpha_factor = targets * self.alpha[1] + (1 - targets) * self.alpha[0]
        loss = alpha_factor * loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LesionDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(
            # start change
            # os.path.join("input", "MyImages", row.image_name + ".jpg")  # original
            os.path.join("/home/anri21/be-fair/aide/MyData/MyImages", row.image_name + ".jpg")
            # end change
        ).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(row.target, dtype=torch.float32)
        return img, label


# Transforms
train_tf = T.Compose(
    [
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(20),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
val_tf = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
# start change
# df = pd.read_csv(os.path.join("input", "mydataset.csv"))  # original
df = pd.read_csv("/home/anri21/be-fair/aide/MyData/mydataset.csv")
# end change
df["target"] = (df["label"] == "malignant").astype(int)
df["skin_tone_group"] = df["skin_tone"].fillna(-1).astype(int)

# 5-fold stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs = []
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df.target)):
    df_train, df_val = df.iloc[train_idx], df.iloc[val_idx]
    # compute weights for fairness
    grp_counts = df_train.skin_tone_group.value_counts().to_dict()
    inv_grp = {g: 1.0 / c for g, c in grp_counts.items()}
    class_counts = df_train.target.value_counts().to_dict()
    inv_cl = {cl: 1.0 / c for cl, c in class_counts.items()}
    w_grp = df_train.skin_tone_group.map(inv_grp).values
    w_cl = df_train.target.map(inv_cl).values
    sample_w = torch.tensor(w_grp * w_cl, dtype=torch.double)
    sampler = WeightedRandomSampler(
        sample_w, num_samples=len(sample_w), replacement=True
    )
    # focal loss alpha
    pos = df_train.target.sum()
    neg = len(df_train) - pos
    alpha_pos = neg / (pos + neg)
    alpha_neg = pos / (pos + neg)
    criterion = FocalLoss(alpha=(alpha_neg, alpha_pos), gamma=2.0).to(device)
    # datasets & loaders
    train_ds = LesionDataset(df_train, train_tf)
    val_ds = LesionDataset(df_val, val_tf)
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
    # model, optimizer, scheduler
    model = timm.create_model(
        "vit_base_patch16_224", pretrained=True, num_classes=1
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    # train
    for epoch in range(3):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs).view(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
    # validation with extended TTA
    model.eval()
    preds, trues = [], []
    flips = [None, [3], [2], [2, 3]]
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            acc = torch.zeros(imgs.size(0), device=device)
            for f in flips:
                inp = imgs if f is None else imgs.flip(dims=f)
                logits = model(inp).view(-1)
                acc += torch.sigmoid(logits)
            acc /= len(flips)
            preds.extend(acc.cpu().numpy())
            trues.extend(labels.numpy())
    auc = roc_auc_score(trues, preds)
    print(f"Fold {fold} AUC: {auc:.4f}")
    aucs.append(auc)

mean_auc = np.mean(aucs)
print(f"Mean CV AUC: {mean_auc:.4f}")

# Retrain on full data
grp_counts = df.skin_tone_group.value_counts().to_dict()
inv_grp = {g: 1.0 / c for g, c in grp_counts.items()}
class_counts = df.target.value_counts().to_dict()
inv_cl = {cl: 1.0 / c for cl, c in class_counts.items()}
w_grp = df.skin_tone_group.map(inv_grp).values
w_cl = df.target.map(inv_cl).values
sample_w = torch.tensor(w_grp * w_cl, dtype=torch.double)
sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
pos = df.target.sum()
neg = len(df) - pos
alpha_pos = neg / (pos + neg)
alpha_neg = pos / (pos + neg)
criterion_full = FocalLoss(alpha=(alpha_neg, alpha_pos), gamma=2.0).to(device)
full_ds = LesionDataset(df, train_tf)
full_loader = DataLoader(full_ds, batch_size=32, sampler=sampler, num_workers=4)
model_full = timm.create_model(
    "vit_base_patch16_224", pretrained=True, num_classes=1
).to(device)
optimizer = optim.AdamW(model_full.parameters(), lr=3e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
for epoch in range(3):
    model_full.train()
    for imgs, labels in full_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model_full(imgs).view(-1)
        loss = criterion_full(logits, labels)
        loss.backward()
        optimizer.step()
    scheduler.step()
os.makedirs("working", exist_ok=True)
torch.save(model_full.state_dict(), os.path.join("working", "model.pth"))
print("Model saved to working/model.pth")


def predict(folder_path):
    """
    Predict malignancy probability for each .jpg in folder_path using multi-flip TTA.
    Returns a DataFrame with columns ['image_name','probability'].
    """
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=1)
    model.load_state_dict(torch.load("working/model.pth", map_location=device))
    model.to(device).eval()
    results = []
    flips = [None, [3], [2], [2, 3]]
    for fname in os.listdir(folder_path):
        # start change
        # if not fname.lower().endswith(".jpg"):  # original
        if not fname.lower().endswith((".jpg", ".png")):
        # end change
            continue
        img = Image.open(os.path.join(folder_path, fname)).convert("RGB")
        inp0 = val_tf(img).unsqueeze(0).to(device)
        prob = 0.0
        with torch.no_grad():
            for f in flips:
                inp = inp0 if f is None else inp0.flip(dims=f)
                logits = model(inp).view(-1)
                prob += torch.sigmoid(logits)
        prob = (prob / len(flips)).item()
        results.append({"image_name": os.path.splitext(fname)[0], "probability": prob})
    return pd.DataFrame(results)


# optional: generate submission if test folder exists
# start change
# test_folder = os.path.join("input", "test_images")  # original
# if os.path.isdir(test_folder):  # original guard
#     submission = predict(test_folder)  # original
#     submission.to_csv(os.path.join("working", "submission.csv"), index=False)  # original
_ddi_files = sorted(os.listdir("/home/anri21/be-fair/evaluation/DDI/images"))
_sub = predict("/home/anri21/be-fair/evaluation/DDI/images")
_pmap = dict(zip(_sub.iloc[:, 0], _sub.iloc[:, 1].astype(float)))
pd.DataFrame({
    "DDI_file": _ddi_files,
    "predicted_probability": [_pmap[os.path.splitext(f)[0]] for f in _ddi_files]
}).to_csv("./DDI_predictions.csv", index=False)
# end change
