import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import warnings

warnings.filterwarnings("ignore")


def mixup_data(x, y, w, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    if batch_size == 1:
        return x, y, w
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    w_a, w_b = w, w[index]
    mixed_y = lam * y_a + (1 - lam) * y_b
    mixed_w = lam * w_a + (1 - lam) * w_b
    return mixed_x, mixed_y, mixed_w


class SkinLesionDataset(Dataset):
    def __init__(self, df, image_dir, transforms):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["image_name"] + ".jpg")
        image = Image.open(img_path).convert("RGB")
        img_t = self.transforms(image)
        label = torch.tensor(row["target"], dtype=torch.float32)
        weight = torch.tensor(row["weight"], dtype=torch.float32)
        return img_t, label, weight


def predict(folder_path: str):
    """
    Predict malignancy probabilities for all .jpg images in folder_path using 10-crop TTA.
    Returns a dict {filename: probability}.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model.load_state_dict(torch.load("./working/model.pth", map_location=device))
    model.to(device).eval()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tx = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.TenCrop(224),
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
    results = {}
    for fname in os.listdir(folder_path):
        # start change
        # if not fname.lower().endswith(".jpg"):  # original
        if not fname.lower().endswith((".jpg", ".png")):
        # end change
            continue
        img = Image.open(os.path.join(folder_path, fname)).convert("RGB")
        crops = tx(img)  # [10,3,224,224]
        with torch.no_grad():
            inputs = crops.to(device)
            logits = model(inputs).squeeze(1)  # [10]
            prob = torch.sigmoid(logits).mean().item()
        results[fname] = prob
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # start change
    # df = pd.read_csv("./input/mydataset.csv")  # original
    df = pd.read_csv("/home/anri21/be-fair/aide/MyData/mydataset.csv")
    # end change
    df["target"] = (df["label"] == "malignant").astype(int)
    freq = df["skin_tone"].value_counts().to_dict()
    df["weight"] = df["skin_tone"].map(lambda x: 1.0 / freq.get(x, 1))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = []
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["target"])):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_tf = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        val_tf = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.TenCrop(224),
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

        # start change
        # train_ds = SkinLesionDataset(train_df, "./input/MyImages", train_tf)  # original
        # val_ds = SkinLesionDataset(val_df, "./input/MyImages", val_tf)  # original
        train_ds = SkinLesionDataset(train_df, "/home/anri21/be-fair/aide/MyData/MyImages", train_tf)
        val_ds = SkinLesionDataset(val_df, "/home/anri21/be-fair/aide/MyData/MyImages", val_tf)
        # end change
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)

        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        bce = nn.BCEWithLogitsLoss(reduction="none")

        model.train()
        for epoch in range(3):
            for imgs, labels, weights in train_loader:
                imgs, labels, weights = (
                    imgs.to(device),
                    labels.to(device),
                    weights.to(device),
                )
                inputs, targets, wts = mixup_data(imgs, labels, weights, alpha=0.4)
                logits = model(inputs).squeeze(1)
                loss = (bce(logits, targets) * wts).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        all_labels, all_probs = [], []
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                B, NC, C, H, W = imgs.shape
                imgs = imgs.view(-1, C, H, W).to(device)
                logits = model(imgs).squeeze(1)
                logits = logits.view(B, NC)
                probs = torch.sigmoid(logits).mean(dim=1).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())
        auc = roc_auc_score(all_labels, all_probs)
        print(f"Fold {fold+1} AUROC: {auc:.4f}")
        cv_aucs.append(auc)

    mean_auc = np.mean(cv_aucs)
    print(f"Mean CV AUROC with EfficientNet-B0: {mean_auc:.4f}")

    # Retrain on full dataset
    full_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    # start change
    # full_ds = SkinLesionDataset(df, "./input/MyImages", full_tf)  # original
    full_ds = SkinLesionDataset(df, "/home/anri21/be-fair/aide/MyData/MyImages", full_tf)
    # end change
    full_loader = DataLoader(full_ds, batch_size=32, shuffle=True, num_workers=4)

    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    model.train()
    for epoch in range(3):
        for imgs, labels, weights in full_loader:
            imgs, labels, weights = (
                imgs.to(device),
                labels.to(device),
                weights.to(device),
            )
            inputs, targets, wts = mixup_data(imgs, labels, weights, alpha=0.4)
            logits = model(inputs).squeeze(1)
            loss = (bce(logits, targets) * wts).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    os.makedirs("./working", exist_ok=True)
    torch.save(model.state_dict(), "./working/model.pth")


if __name__ == "__main__":
    main()
    # start change
    _ddi_files = sorted(os.listdir("/home/anri21/be-fair/evaluation/DDI/images"))
    _preds = predict("/home/anri21/be-fair/evaluation/DDI/images")
    pd.DataFrame({
        "DDI_file": _ddi_files,
        "predicted_probability": [float(_preds[f]) for f in _ddi_files]
    }).to_csv("./DDI_predictions.csv", index=False)
    # end change
