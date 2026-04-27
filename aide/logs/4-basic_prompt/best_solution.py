import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models


class LesionDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
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
        label = float(row["label_bin"])
        return image, label


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        lbls = lbls.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


def eval_model(model, loader, device):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            out1 = model(imgs)
            out2 = model(torch.flip(imgs, dims=[3]))
            avg_out = (out1 + out2) / 2
            probs = torch.sigmoid(avg_out).cpu().numpy().flatten()
            preds.extend(probs)
            truths.extend(lbls.numpy().flatten())
    return roc_auc_score(truths, preds)


def predict(folder_path):
    """
    Load the five fold DenseNet-161 models and compute malignancy probabilities
    using horizontal-flip TTA. Returns a pandas DataFrame with
    ['image_name','malignancy_probability'].
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prepare models
    models_list = []
    for fold in range(5):
        model = models.densenet161(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
        state_path = os.path.join("working", f"densenet161_fold{fold}.pth")
        model.load_state_dict(torch.load(state_path, map_location=device))
        model.to(device).eval()
        models_list.append(model)
    # Transform
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    results = []
    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img = Image.open(os.path.join(folder_path, fname)).convert("RGB")
        inp = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            sum_prob = 0.0
            for model in models_list:
                o1 = model(inp)
                o2 = model(torch.flip(inp, dims=[3]))
                prob = torch.sigmoid((o1 + o2) / 2).item()
                sum_prob += prob
            avg_prob = sum_prob / len(models_list)
        results.append({"image_name": fname, "malignancy_probability": avg_prob})
    df = pd.DataFrame(results)
    # If test submission is needed:
    submission_path = os.path.join("working", "submission.csv")
    df.to_csv(submission_path, index=False)
    return df


if __name__ == "__main__":
    # Paths
    csv_path = os.path.join("input", "mydataset.csv")
    img_dir = os.path.join("input", "MyImages")
    df = pd.read_csv(csv_path)
    df["label_bin"] = (df["label"] == "malignant").astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    os.makedirs("working", exist_ok=True)
    aucs = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label_bin"])):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        train_ds = LesionDataset(train_df, img_dir, train_transform)
        val_ds = LesionDataset(val_df, img_dir, val_transform)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

        model = models.densenet161(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)

        for epoch in range(3):
            train_one_epoch(model, train_loader, criterion, optimizer, device)
            scheduler.step()

        auc = eval_model(model, val_loader, device)
        print(f"Fold {fold+1} ROC AUC: {auc:.4f}")
        aucs.append(auc)
        # Save fold model
        torch.save(
            model.state_dict(), os.path.join("working", f"densenet161_fold{fold}.pth")
        )

    mean_auc, std_auc = np.mean(aucs), np.std(aucs)
    print(f"Mean CV ROC AUC: {mean_auc:.4f} ± {std_auc:.4f}")
