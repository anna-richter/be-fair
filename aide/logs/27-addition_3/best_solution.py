import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SkinLesionDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.labels = (self.df["label"] == "malignant").astype(float).values
        self.names = self.df["image_name"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.names[idx] + ".jpg"
        path = os.path.join(self.img_dir, img_name)
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label, self.names[idx]


def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv("./input/mydataset.csv")
    df = df[
        df["skin_tone"].notnull() & df["image_name"].notnull() & df["label"].notnull()
    ]
    freq = df["skin_tone"].value_counts().to_dict()
    df["weight"] = df["skin_tone"].map(lambda x: 1.0 / freq.get(x, 1)).astype(float)

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=(df["label"] == "malignant"), random_state=42
    )

    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.5),
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

    train_ds = SkinLesionDataset(train_df, "./input/MyImages", transform=train_tf)
    val_ds = SkinLesionDataset(val_df, "./input/MyImages", transform=val_tf)

    sampler = WeightedRandomSampler(
        weights=train_df["weight"].values, num_samples=len(train_df), replacement=True
    )
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)

    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    model = model.to(device)

    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_auc = 0.0
    os.makedirs("./working", exist_ok=True)

    for epoch in range(1, 6):
        model.train()
        running_loss = 0.0
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs = imgs.to(device)
                log_o = model(imgs).squeeze(1)
                log_h = model(torch.flip(imgs, dims=[3])).squeeze(1)
                log_v = model(torch.flip(imgs, dims=[2])).squeeze(1)
                log_hv = model(torch.flip(imgs, dims=[2, 3])).squeeze(1)
                avg_logits = (log_o + log_h + log_v + log_hv) / 4
                probs = torch.sigmoid(avg_logits).cpu().numpy()
                ys.extend(labels.numpy())
                ps.extend(probs)
        auc = roc_auc_score(ys, ps)
        print(f"Epoch {epoch} - Val AUROC (4-way TTA): {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "./working/model.pth")

    print(f"Best Val AUROC: {best_auc:.4f}")
    return best_auc


def predict(folder_path):
    """
    Predict malignancy probabilities with 4-way flip TTA.
    Args:
        folder_path (str): Path containing .jpg images.
    Returns:
        pd.DataFrame: columns ['image_name','probability'] with averaged 4-way TTA probs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    model.load_state_dict(torch.load("./working/model.pth", map_location=device))
    model = model.to(device).eval()

    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    results = []
    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith(".jpg"):
            continue
        img = Image.open(os.path.join(folder_path, fname)).convert("RGB")
        img_t = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            log_o = model(img_t).squeeze(1)
            log_h = model(torch.flip(img_t, dims=[3])).squeeze(1)
            log_v = model(torch.flip(img_t, dims=[2])).squeeze(1)
            log_hv = model(torch.flip(img_t, dims=[2, 3])).squeeze(1)
            prob = torch.sigmoid((log_o + log_h + log_v + log_hv) / 4).item()
        results.append({"image_name": fname, "probability": prob})
    return pd.DataFrame(results)


if __name__ == "__main__":
    best_auc = train_and_evaluate()
    preds = predict("./input/MyImages")
    preds.to_csv("./working/submission.csv", index=False)
    print(f"Best Val AUROC: {best_auc:.4f}")
    print(f"Saved submission.csv with {len(preds)} predictions")
