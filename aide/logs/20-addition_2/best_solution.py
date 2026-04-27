import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models


# MixUp helper
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# Dataset definition
class LesionDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_name"] + ".jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = row["binary_label"]
        skin_tone = row["skin_tone"]
        return image, label, skin_tone


# Predict function
def predict(model_path, image_folder, batch_size=32):
    """
    Load trained model and predict malignancy probability for each image in image_folder.
    Returns dict {image_name: probability}.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    filenames = sorted(
        [f for f in os.listdir(image_folder) if f.lower().endswith(".jpg")]
    )
    out = {}
    loader = DataLoader(filenames, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            imgs = []
            for fn in batch:
                img = Image.open(os.path.join(image_folder, fn)).convert("RGB")
                imgs.append(transform(img))
            imgs = torch.stack(imgs).to(device)
            # TTA: original + horizontal flip
            logits1 = model(imgs)
            logits2 = model(torch.flip(imgs, dims=[3]))
            probs = (torch.sigmoid(logits1) + torch.sigmoid(logits2)) / 2.0
            probs = probs.cpu().numpy().flatten()
            for fn, p in zip(batch, probs):
                out[fn] = float(p)
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv("./input/mydataset.csv")
    df["binary_label"] = df["label"].map(
        {"malignant": 1, "benign": 0, "non-neoplastic": 0}
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    overall_aucs, gaps = [], []
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["binary_label"]), 1):
        df_train, df_val = df.iloc[train_idx], df.iloc[val_idx]
        tones = df_train["skin_tone"].values
        unique, counts = np.unique(tones, return_counts=True)
        tone_counts = dict(zip(unique, counts))
        weights = [1.0 / tone_counts[t] for t in tones]
        sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )
        train_ds = LesionDataset(df_train, "./input/MyImages", transform)
        val_ds = LesionDataset(df_val, "./input/MyImages", transform)
        train_loader = DataLoader(
            train_ds, batch_size=32, sampler=sampler, num_workers=2
        )
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 1)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        # Training with MixUp
        for epoch in range(5):
            model.train()
            for imgs, labels, _ in train_loader:
                imgs = imgs.to(device)
                labels = labels.float().to(device)
                mixed_imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=0.2)
                preds = model(mixed_imgs).squeeze(1)
                loss = lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # Validation with horizontal-flip TTA
        model.eval()
        ys, ps, tones_val = [], [], []
        with torch.no_grad():
            for imgs, labels, tones in val_loader:
                imgs = imgs.to(device)
                logits1 = model(imgs)
                logits2 = model(torch.flip(imgs, dims=[3]))
                probs = (torch.sigmoid(logits1) + torch.sigmoid(logits2)) / 2.0
                probs = probs.cpu().numpy().flatten()
                ys.extend(labels.numpy())
                ps.extend(probs)
                tones_val.extend(tones.numpy())
        auc = roc_auc_score(ys, ps)
        df_v = pd.DataFrame({"y": ys, "p": ps, "tone": tones_val})
        light = df_v[df_v.tone <= 3]
        dark = df_v[df_v.tone >= 5]
        auc_light = roc_auc_score(light.y, light.p) if len(light) > 0 else np.nan
        auc_dark = roc_auc_score(dark.y, dark.p) if len(dark) > 0 else np.nan
        gap = abs(auc_light - auc_dark)
        overall_aucs.append(auc)
        gaps.append(gap)
        print(f"Fold {fold}: AUROC={auc:.4f}, gap={gap:.4f}")
    print(f"Mean CV AUROC: {np.nanmean(overall_aucs):.4f}")
    print(f"Mean fairness gap (light vs dark): {np.nanmean(gaps):.4f}")
    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    main()
