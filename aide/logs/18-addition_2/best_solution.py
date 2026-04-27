import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


class LesionDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row.image_name + ".jpg")).convert(
            "RGB"
        )
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(row.y, dtype=torch.float32)
        return img, label


def predict(image_folder, model_path="./working/model_densenet121.pth", batch_size=32):
    """
    Predict malignancy probabilities with horizontal-flip TTA.
    Returns DataFrame with ['image_name','probability'].
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    results = []
    imgs = [f for f in sorted(os.listdir(image_folder)) if f.lower().endswith(".jpg")]
    for i in range(0, len(imgs), batch_size):
        batch = imgs[i : i + batch_size]
        tensors = []
        for fname in batch:
            img = Image.open(os.path.join(image_folder, fname)).convert("RGB")
            tensors.append(tf(img))
        x = torch.stack(tensors).to(device)
        with torch.no_grad():
            logits1 = model(x)
            logits2 = model(torch.flip(x, dims=[3]))
            probs = torch.sigmoid((logits1 + logits2) / 2).cpu().numpy().flatten()
        for fname, p in zip(batch, probs):
            results.append({"image_name": fname, "probability": p})
    return pd.DataFrame(results)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    df = pd.read_csv("./input/mydataset.csv")
    df = df[df.label.isin(["benign", "malignant"])].copy()
    df["y"] = (df.label == "malignant").astype(int)
    if "skin_tone" not in df.columns:
        raise ValueError("Dataset must contain a 'skin_tone' column.")

    def tone_cat(x):
        if x in [1, 2, 3]:
            return "light"
        elif x in [5, 6]:
            return "dark"
        else:
            return "medium"

    df["tone_cat"] = df.skin_tone.apply(tone_cat)
    df["stratify"] = df.y.astype(str) + "_" + df.tone_cat
    img_dir = "./input/MyImages"
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    overall_aucs, light_aucs, dark_aucs = [], [], []
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df.stratify)):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        train_ds = LesionDataset(train_df, img_dir, transform=train_tf)
        val_ds = LesionDataset(val_df, img_dir, transform=val_tf)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
        criterion = nn.BCEWithLogitsLoss()
        for epoch in range(3):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device).unsqueeze(1)
                loss = criterion(model(x), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits1 = model(x)
                logits2 = model(torch.flip(x, dims=[3]))
                probs = torch.sigmoid((logits1 + logits2) / 2).cpu().numpy().flatten()
                all_probs.extend(probs)
                all_labels.extend(y.numpy().flatten())
        tones = val_df.tone_cat.values
        auc_all = roc_auc_score(all_labels, all_probs)
        light_mask = tones == "light"
        dark_mask = tones == "dark"
        auc_light = (
            roc_auc_score(
                np.array(all_labels)[light_mask], np.array(all_probs)[light_mask]
            )
            if light_mask.sum() > 0
            else np.nan
        )
        auc_dark = (
            roc_auc_score(
                np.array(all_labels)[dark_mask], np.array(all_probs)[dark_mask]
            )
            if dark_mask.sum() > 0
            else np.nan
        )
        gap = (
            abs(auc_light - auc_dark)
            if not np.isnan(auc_light) and not np.isnan(auc_dark)
            else np.nan
        )
        print(
            f"Fold {fold+1} AUROC: overall={auc_all:.4f}, light={auc_light:.4f}, dark={auc_dark:.4f}, gap={gap:.4f}"
        )
        overall_aucs.append(auc_all)
        light_aucs.append(auc_light)
        dark_aucs.append(auc_dark)
    print(
        f"Mean AUROC: overall={np.nanmean(overall_aucs):.4f}±{np.nanstd(overall_aucs):.4f}, light={np.nanmean(light_aucs):.4f}, dark={np.nanmean(dark_aucs):.4f}, gap={abs(np.nanmean(light_aucs)-np.nanmean(dark_aucs)):.4f}"
    )
    # Retrain on full data
    full_ds = LesionDataset(df, img_dir, transform=train_tf)
    full_loader = DataLoader(full_ds, batch_size=32, shuffle=True, num_workers=4)
    final_model = models.densenet121(pretrained=True)
    final_model.classifier = nn.Linear(final_model.classifier.in_features, 1)
    final_model.to(device)
    opt = torch.optim.AdamW(final_model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched_full = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3)
    crit = nn.BCEWithLogitsLoss()
    for epoch in range(3):
        final_model.train()
        for x, y in full_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            l = crit(final_model(x), y)
            opt.zero_grad()
            l.backward()
            opt.step()
        sched_full.step()
    os.makedirs("./working", exist_ok=True)
    save_path = "./working/model_densenet121.pth"
    torch.save(final_model.state_dict(), save_path)
    print(f"Final model saved to {save_path}")
    test_folder = "./input/test_images"
    if os.path.isdir(test_folder):
        sub = predict(test_folder, model_path=save_path)
        os.makedirs("./working", exist_ok=True)
        sub.to_csv("./working/submission.csv", index=False)
        print("Submission saved to ./working/submission.csv")


if __name__ == "__main__":
    main()
