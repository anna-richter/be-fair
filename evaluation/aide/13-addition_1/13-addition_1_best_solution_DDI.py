import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


class SkinDataset(Dataset):
    def __init__(self, df, image_dir, transform):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["image_name"] + ".jpg")
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(row["label_bin"], dtype=torch.float32)
        return x, y


def predict(folder_path):
    """
    Predict malignancy probabilities for images in folder_path using TTA TenCrop.
    Returns a dict mapping filename to probability.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.densenet121(pretrained=False)
    # Dropout before final layer to match training
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5), nn.Linear(model.classifier.in_features, 1)
    )
    model.load_state_dict(torch.load("working/best_model.pth", map_location=device))
    model.to(device).eval()

    resize = transforms.Resize(256)
    tencrop = transforms.TenCrop(224)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    probs = {}
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith((".jpg", "jpeg", "png")):
            continue
        img = Image.open(os.path.join(folder_path, fname)).convert("RGB")
        crops = tencrop(resize(img))
        batch = torch.stack([normalize(to_tensor(c)) for c in crops]).to(device)
        with torch.no_grad():
            out = model(batch)
            p = torch.sigmoid(out).mean().item()
        probs[fname] = p
    return probs


def main():
    # config
    # start change
    # image_dir = "input/MyImages"  # original
    image_dir = "/home/anri21/be-fair/aide/MyData/MyImages"
    # end change
    # start change
    # df = pd.read_csv("input/mydataset.csv")  # original
    df = pd.read_csv("/home/anri21/be-fair/aide/MyData/mydataset.csv")
    # end change
    df["label_bin"] = (df["label"] == "malignant").astype(int)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    # TTA transform for validation
    resize = transforms.Resize(256)
    tencrop = transforms.TenCrop(224)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean, std)

    def val_transform(img):
        crops = tencrop(resize(img))
        return torch.stack([normalize(to_tensor(c)) for c in crops])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X = df["image_name"].values
    y = df["label_bin"].values
    cv_aucs = []
    os.makedirs("working", exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        # weighted sampler for train
        counts = train_df["label_bin"].value_counts().to_dict()
        weights = train_df["label_bin"].apply(lambda x: 1.0 / counts[x]).values
        sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )

        train_ds = SkinDataset(train_df, image_dir, transform=train_transform)
        val_ds = SkinDataset(val_df, image_dir, transform=val_transform)
        train_loader = DataLoader(
            train_ds, batch_size=32, sampler=sampler, num_workers=4
        )
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)

        model = models.densenet121(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(model.classifier.in_features, 1)
        )
        model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        best_auc = 0.0
        for epoch in range(1, 6):
            model.train()
            for imgs, labels in train_loader:
                imgs = imgs.to(device)
                labels = labels.unsqueeze(1).to(device)
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()

            model.eval()
            all_labels, all_probs = [], []
            with torch.no_grad():
                for imgs, labels in val_loader:
                    bs, nc, C, H, W = imgs.size()
                    imgs = imgs.view(-1, C, H, W).to(device)
                    out = model(imgs)
                    out = out.view(bs, nc, -1)
                    probs = torch.sigmoid(out).mean(dim=1)
                    all_labels += labels.numpy().tolist()
                    all_probs += probs.cpu().numpy().ravel().tolist()
            auc = roc_auc_score(all_labels, all_probs)
            if auc > best_auc:
                best_auc = auc
            print(f"Fold {fold} Epoch {epoch}: val AUROC={auc:.4f}")
        print(f"Fold {fold} best AUROC: {best_auc:.4f}")
        cv_aucs.append(best_auc)

    print(f"CV Mean AUROC: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")

    # Final training on full data
    counts_full = df["label_bin"].value_counts().to_dict()
    weights_full = df["label_bin"].apply(lambda x: 1.0 / counts_full[x]).values
    sampler_full = WeightedRandomSampler(
        weights_full, num_samples=len(weights_full), replacement=True
    )
    full_ds = SkinDataset(df, image_dir, transform=train_transform)
    full_loader = DataLoader(
        full_ds, batch_size=32, sampler=sampler_full, num_workers=4
    )

    model_full = models.densenet121(pretrained=True)
    model_full.classifier = nn.Sequential(
        nn.Dropout(p=0.5), nn.Linear(model_full.classifier.in_features, 1)
    )
    model_full.to(device)
    criterion_full = nn.BCEWithLogitsLoss()
    optimizer_full = optim.Adam(model_full.parameters(), lr=1e-4)

    for epoch in range(1, 6):
        model_full.train()
        for imgs, labels in full_loader:
            imgs = imgs.to(device)
            labels = labels.unsqueeze(1).to(device)
            optimizer_full.zero_grad()
            out = model_full(imgs)
            loss = criterion_full(out, labels)
            loss.backward()
            optimizer_full.step()
        print(f"Full data training epoch {epoch} complete")
    torch.save(model_full.state_dict(), "working/best_model.pth")
    print("Final model trained on full data and saved to working/best_model.pth")
    # start change
    _ddi_files = sorted(os.listdir("/home/anri21/be-fair/evaluation/DDI/images"))
    _preds = predict("/home/anri21/be-fair/evaluation/DDI/images")
    pd.DataFrame({
        "DDI_file": _ddi_files,
        "predicted_probability": [float(_preds[f]) for f in _ddi_files]
    }).to_csv("./DDI_predictions.csv", index=False)
    # end change


if __name__ == "__main__":
    main()
