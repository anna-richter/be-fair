import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


class SkinLesionDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row.image_name + ".jpg")
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(row.label_bin, dtype=torch.float32)
        return img, label


class TestFolderDataset(Dataset):
    def __init__(self, folder, transform):
        # start change
        # self.files = [f for f in os.listdir(folder) if f.lower().endswith(".jpg")]  # original
        self.files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))]
        # end change
        self.folder = folder
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = Image.open(os.path.join(self.folder, fname)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, fname


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            # original
            logits1 = model(imgs).squeeze(1)
            # horizontal flip TTA
            imgs_flipped = torch.flip(imgs, dims=[3])
            logits2 = model(imgs_flipped).squeeze(1)
            probs = (torch.sigmoid(logits1) + torch.sigmoid(logits2)) / 2.0
            ys.extend(labels.numpy())
            ps.extend(probs.cpu().numpy())
    return roc_auc_score(ys, ps)


def predict(image_folder: str):
    """
    Predict malignancy probabilities for all .jpg images in image_folder with TTA.
    Returns a DataFrame with columns ['image_name','malignancy_probability'].
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    model.load_state_dict(torch.load("./working/model.pth", map_location=device))
    model.to(device).eval()
    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    ds = TestFolderDataset(image_folder, tf)
    loader = DataLoader(ds, batch_size=32, num_workers=4, pin_memory=True)
    results = []
    with torch.no_grad():
        for imgs, fnames in loader:
            imgs = imgs.to(device)
            logits1 = model(imgs).squeeze(1)
            imgs_f = torch.flip(imgs, dims=[3])
            logits2 = model(imgs_f).squeeze(1)
            probs = (
                ((torch.sigmoid(logits1) + torch.sigmoid(logits2)) / 2.0).cpu().numpy()
            )
            for f, p in zip(fnames, probs):
                results.append((f, float(p)))
    return pd.DataFrame(results, columns=["image_name", "malignancy_probability"])


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # start change
    # df = pd.read_csv("./input/mydataset.csv")  # original
    df = pd.read_csv("/home/anri21/be-fair/aide/MyData/mydataset.csv")
    # end change
    df["label_bin"] = (df.label == "malignant").astype(int)
    # start change
    # img_dir = "./input/MyImages"  # original
    img_dir = "/home/anri21/be-fair/aide/MyData/MyImages"
    # end change
    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
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
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, df.label_bin), 1):
        train_df, val_df = df.iloc[tr_idx], df.iloc[va_idx]
        train_ds = SkinLesionDataset(train_df, img_dir, train_tf)
        val_ds = SkinLesionDataset(val_df, img_dir, val_tf)
        train_loader = DataLoader(
            train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
        )
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
        model = model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        for epoch in range(3):
            train_one_epoch(model, train_loader, criterion, optimizer, device)
        auc = evaluate(model, val_loader, device)
        print(f"Fold {fold} AUROC with TTA: {auc:.4f}")
        aucs.append(auc)
    mean_auc = np.mean(aucs)
    print(f"Mean CV AUROC with TTA: {mean_auc:.4f}")
    # Retrain on full data
    full_ds = SkinLesionDataset(df, img_dir, train_tf)
    full_loader = DataLoader(
        full_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(3):
        train_one_epoch(model, full_loader, criterion, optimizer, device)
    os.makedirs("./working", exist_ok=True)
    torch.save(model.state_dict(), "./working/model.pth")
    # start change
    # test_folder = "./input/test"  # original
    # if os.path.isdir(test_folder):  # original guard
    #     sub = predict(test_folder)  # original
    #     sub.to_csv("./working/submission.csv", index=False)  # original
    _ddi_files = sorted(os.listdir("/home/anri21/be-fair/evaluation/DDI/images"))
    _sub = predict("/home/anri21/be-fair/evaluation/DDI/images")
    _pmap = dict(zip(_sub.iloc[:, 0], _sub.iloc[:, 1].astype(float)))
    pd.DataFrame({
        "DDI_file": _ddi_files,
        "predicted_probability": [_pmap[f] for f in _ddi_files]
    }).to_csv("./DDI_predictions.csv", index=False)
    # end change
