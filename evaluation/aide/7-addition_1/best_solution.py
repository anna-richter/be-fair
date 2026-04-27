import os
import numpy as np
import torch
import pandas as pd
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


class SkinDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.label_map = {"malignant": 1.0, "benign": 0.0, "non-neoplastic": 0.0}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        path = os.path.join(self.img_dir, row.image_name + ".jpg")
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.label_map[row.label], dtype=torch.float32)
        return img, label


def mixup_data(x, y, alpha=0.4, device="cpu"):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def predict(folder_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b2(pretrained=False)
    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, 1)
    model.load_state_dict(torch.load("working/best_model.pth", map_location=device))
    model.to(device).eval()
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    ts = transforms.ToTensor()
    norm = transforms.Normalize(mean, std)
    tta = transforms.Compose(
        [
            transforms.Resize(280),
            transforms.TenCrop(260),
            transforms.Lambda(lambda crops: torch.stack([norm(ts(c)) for c in crops])),
        ]
    )
    records = []
    with torch.no_grad():
        for fname in sorted(os.listdir(folder_path)):
            if not fname.lower().endswith(".jpg"):
                continue
            img = Image.open(os.path.join(folder_path, fname)).convert("RGB")
            crops = tta(img).to(device)  # (10, C, H, W)
            logits = model(crops).squeeze(1)
            prob = torch.sigmoid(logits).mean().item()
            records.append({"image_name": fname, "malignancy_prob": prob})
    return pd.DataFrame(records)


def main():
    df = pd.read_csv("input/mydataset.csv")
    img_dir = os.path.join("input", "MyImages")
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df.label, random_state=42
    )
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(260),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(280),
            transforms.TenCrop(260),
            transforms.Lambda(
                lambda crops: torch.stack(
                    [
                        transforms.Normalize(mean, std)(transforms.ToTensor()(c))
                        for c in crops
                    ]
                )
            ),
        ]
    )
    train_ds = SkinDataset(train_df, img_dir, transform=train_tf)
    val_ds = SkinDataset(val_df, img_dir, transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b2(pretrained=True)
    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, 1)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_auc = 0.0
    os.makedirs("working", exist_ok=True)

    for epoch in range(8):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            mixed_imgs, y_a, y_b, lam = mixup_data(
                imgs, labels, alpha=0.4, device=device
            )
            optimizer.zero_grad()
            outputs = model(mixed_imgs).squeeze(1)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            loss.backward()
            optimizer.step()

        model.eval()
        all_labels, all_probs = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                b, tta, C, H, W = imgs.shape
                imgs = imgs.view(-1, C, H, W).to(device)
                logits = model(imgs).view(b, tta)
                probs = torch.sigmoid(logits).mean(1).cpu().tolist()
                all_probs.extend(probs)
                all_labels.extend(labels.tolist())

        auc = roc_auc_score(all_labels, all_probs)
        print(f"Epoch {epoch+1}/8 - Val AUROC: {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "working/best_model.pth")

    print(f"Best Validation AUROC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
