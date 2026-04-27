import os
import copy
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


class SkinLesionDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_name"] + ".jpg")
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(row["binary_label"], dtype=torch.float32)
        return img, label


def cutmix_data(x, y, alpha=1.0, device="cpu"):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size).to(device)
    y_a, y_b = y, y[index]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    x_cut = x.clone()
    x_cut[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    lam_adjusted = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return x_cut, y_a, y_b, lam_adjusted


def main():
    # reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv("input/mydataset.csv")
    df["binary_label"] = (df["label"] == "malignant").astype(int)
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["binary_label"], random_state=seed
    )

    # Transforms
    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Datasets
    train_ds = SkinLesionDataset(train_df, "input/MyImages", train_transforms)
    val_ds = SkinLesionDataset(val_df, "input/MyImages", val_transforms)

    # Weighted sampler to balance classes
    labels = train_df["binary_label"].values
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_ds, batch_size=32, sampler=sampler, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    # Model and EMA
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    model = model.to(device)
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)
    ema_model.to(device)
    ema_decay = 0.999

    # Loss, optimizer, scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
    epochs = 8

    # Training loop
    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            mixed_imgs, y_a, y_b, lam = cutmix_data(
                imgs, labels, alpha=1.0, device=device
            )
            optimizer.zero_grad()
            outputs = model(mixed_imgs).squeeze(1)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            loss.backward()
            optimizer.step()
            # EMA update
            with torch.no_grad():
                msd = model.state_dict()
                for k, v in ema_model.state_dict().items():
                    v.copy_(ema_decay * v + (1.0 - ema_decay) * msd[k])
        scheduler.step()

    # Validation with EMA + TTA
    ema_model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            out_orig = ema_model(imgs).squeeze(1)
            out_h = ema_model(torch.flip(imgs, dims=[3])).squeeze(1)
            out_v = ema_model(torch.flip(imgs, dims=[2])).squeeze(1)
            probs = (
                torch.sigmoid(out_orig) + torch.sigmoid(out_h) + torch.sigmoid(out_v)
            ) / 3.0
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    auc = roc_auc_score(all_labels, all_preds)
    print(f"Validation AUROC with WeightedSampler + EMA + CutMix + TTA: {auc:.4f}")

    os.makedirs("working", exist_ok=True)
    torch.save(ema_model.state_dict(), "working/model.pth")


def predict(folder_path):
    """
    Load the trained EMA model and return a dict mapping image filenames to malignancy probability.
    Args:
        folder_path (str): Path to a folder containing .jpg images.
    Returns:
        dict: {filename: probability}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    model.load_state_dict(torch.load("working/model.pth", map_location=device))
    model = model.to(device).eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    results = {}
    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(folder_path, fname)
        img = Image.open(path).convert("RGB")
        inp = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out1 = torch.sigmoid(model(inp).squeeze(1))
            out2 = torch.sigmoid(model(torch.flip(inp, dims=[3])).squeeze(1))
            out3 = torch.sigmoid(model(torch.flip(inp, dims=[2])).squeeze(1))
            prob = ((out1 + out2 + out3) / 3.0).item()
        results[fname] = prob
    return results


if __name__ == "__main__":
    main()
