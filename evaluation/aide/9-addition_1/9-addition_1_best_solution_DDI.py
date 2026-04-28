import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import timm


class SkinLesionDataset(Dataset):
    def __init__(self, df, transform=None, return_index=False):
        self.df = df
        self.transform = transform
        self.return_index = return_index

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.filepath).convert("RGB")
        img = self.transform(img) if self.transform else img
        label = torch.tensor(row.target, dtype=torch.float32)
        if self.return_index:
            return img, label, row.name
        return img, label


def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def predict(image_folder: str):
    """
    Load ensemble of 5 models and predict malignancy probabilities with horizontal-flip TTA.
    Args:
        image_folder (str): path to folder containing images.
    Returns:
        Dict[str, float]: mapping from image filename to malignancy probability.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    # Load ensemble
    models = []
    for fold in range(5):
        m = timm.create_model(
            "swin_tiny_patch4_window7_224", pretrained=False, num_classes=1
        )
        ckpt = f"working/model_fold{fold}.pth"
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.to(device).eval()
        models.append(m)
    results = {}
    for fname in sorted(os.listdir(image_folder)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img = Image.open(os.path.join(image_folder, fname)).convert("RGB")
        t = transform(img).unsqueeze(0).to(device)
        t_flipped = torch.flip(t, dims=[3])
        probs = []
        with torch.no_grad():
            for m in models:
                logit1 = m(t).item()
                logit2 = m(t_flipped).item()
                avg_logit = 0.5 * (logit1 + logit2)
                probs.append(1 / (1 + np.exp(-avg_logit)))
        results[fname] = float(np.mean(probs))
    return results


def main():
    # start change
    # df = pd.read_csv("input/mydataset.csv")  # original
    df = pd.read_csv("/home/anri21/be-fair/aide/MyData/mydataset.csv")
    # end change
    df["target"] = (df.label == "malignant").astype(int)
    # start change
    # df["filepath"] = df.image_name.apply(lambda x: os.path.join("input/MyImages", x + ".jpg"))  # original
    df["filepath"] = df.image_name.apply(
        lambda x: os.path.join("/home/anri21/be-fair/aide/MyData/MyImages", x + ".jpg")
    )
    # end change

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(df))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df.target)):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        # Class-balanced sampler
        counts = train_df.target.value_counts().to_dict()
        weights = {cls: 1.0 / count for cls, count in counts.items()}
        sample_weights = train_df.target.map(weights).values
        sampler = WeightedRandomSampler(
            torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )

        train_ds = SkinLesionDataset(train_df, transform=train_transform)
        val_ds = SkinLesionDataset(val_df, transform=val_transform, return_index=True)
        train_loader = DataLoader(
            train_ds, batch_size=32, sampler=sampler, num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
        )

        model = timm.create_model(
            "swin_tiny_patch4_window7_224", pretrained=True, num_classes=1
        )
        model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

        # Training
        for epoch in range(1, 6):
            model.train()
            running_loss = 0.0
            for imgs, targets in train_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                mixed_imgs, y_a, y_b, lam = mixup_data(imgs, targets)
                optimizer.zero_grad()
                outputs = model(mixed_imgs).squeeze(1)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(
                    outputs, y_b
                )
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * imgs.size(0)
            scheduler.step()
            print(f"Fold {fold} Epoch {epoch} - Loss: {running_loss/len(train_ds):.4f}")

        # Validation with horizontal-flip TTA
        model.eval()
        with torch.no_grad():
            for imgs, targets, idxs in val_loader:
                imgs = imgs.to(device)
                imgs_flipped = torch.flip(imgs, dims=[3])
                logits1 = model(imgs).squeeze(1)
                logits2 = model(imgs_flipped).squeeze(1)
                logits = 0.5 * (logits1 + logits2)
                probs = torch.sigmoid(logits).cpu().numpy()
                oof_preds[idxs.numpy()] = probs
        fold_auc = roc_auc_score(df.target.iloc[val_idx], oof_preds[val_idx])
        print(f"Fold {fold} ROC AUC: {fold_auc:.4f}")
        os.makedirs("working", exist_ok=True)
        torch.save(model.state_dict(), f"working/model_fold{fold}.pth")

    overall_auc = roc_auc_score(df.target.values, oof_preds)
    print(f"Overall OOF ROC AUC: {overall_auc:.4f}")


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
