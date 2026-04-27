import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import efficientnet_b0
from torch.optim.lr_scheduler import OneCycleLR

DATA_CSV = "./input/mydataset.csv"
IMG_DIR = "./input/MyImages"
TEST_DIR = "./input/test_images"
WORKING_DIR = "./working"
MODEL_PATH = os.path.join(WORKING_DIR, "model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 5
FOLDS = 5


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        probas = torch.sigmoid(inputs)
        pt = targets * probas + (1 - targets) * (1 - probas)
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_term = (1 - pt) ** self.gamma
        loss = alpha_factor * focal_term * bce
        return loss.mean() if self.reduction == "mean" else loss.sum()


class SkinLesionDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img = Image.open(os.path.join(self.img_dir, row.image_name + ".jpg")).convert(
            "RGB"
        )
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(row.binary_label, dtype=torch.float32)
        return img, label, row.skin_tone, row.image_name


def get_model():
    model = efficientnet_b0(pretrained=True)
    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, 1)
    return model.to(DEVICE)


def train_one_epoch(model, loader, criterion, optimizer, scheduler=None):
    model.train()
    total_loss = 0.0
    for imgs, labels, _, _ in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


def evaluate_with_tta(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for imgs, labels, _, _ in loader:
            imgs = imgs.to(DEVICE)
            p1 = torch.sigmoid(model(imgs))
            p2 = torch.sigmoid(model(torch.flip(imgs, [-1])))
            probs = ((p1 + p2) / 2).cpu().numpy().flatten()
            preds.extend(probs)
            trues.extend(labels.numpy().flatten())
    return roc_auc_score(trues, preds)


def predict(model_path, image_folder, batch_size=32):
    """
    Load one or more trained models (if model_path is a directory containing model_fold*.pth)
    and predict malignancy probability for all images in image_folder by ensembling TTA outputs.
    Returns a dict {image_name: probability}.
    """
    # Get list of model files
    if os.path.isdir(model_path):
        model_files = sorted(glob.glob(os.path.join(model_path, "model_fold*.pth")))
    else:
        model_files = [model_path]
    if not model_files:
        raise ValueError("No model files found at provided model_path.")
    # Preload transforms
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # Load images
    imgs, names = [], []
    for fname in os.listdir(image_folder):
        if fname.lower().endswith((".jpg", ".png")):
            img = Image.open(os.path.join(image_folder, fname)).convert("RGB")
            imgs.append(transform(img))
            names.append(os.path.splitext(fname)[0])
    if not imgs:
        return {}
    dataset = torch.stack(imgs)
    # Ensemble predictions
    agg_probs = np.zeros(len(dataset), dtype=np.float32)
    for mf in model_files:
        model = get_model()
        model.load_state_dict(torch.load(mf, map_location=DEVICE))
        model.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i : i + batch_size].to(DEVICE)
                p1 = torch.sigmoid(model(batch))
                p2 = torch.sigmoid(model(torch.flip(batch, [-1])))
                probs.extend(((p1 + p2) / 2).cpu().numpy().flatten())
        agg_probs += np.array(probs, dtype=np.float32)
    agg_probs /= len(model_files)
    return dict(zip(names, agg_probs.tolist()))


if __name__ == "__main__":
    os.makedirs(WORKING_DIR, exist_ok=True)
    df = pd.read_csv(DATA_CSV)
    df["binary_label"] = (df.label == "malignant").astype(int)
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

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

    fold_aucs = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df.binary_label)):
        train_df, val_df = df.loc[train_idx], df.loc[val_idx]
        freqs = train_df.skin_tone.value_counts().to_dict()
        weights = train_df.skin_tone.map(lambda x: 1.0 / freqs[x]).values
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

        train_ds = SkinLesionDataset(train_df, IMG_DIR, transform=train_tf)
        val_ds = SkinLesionDataset(val_df, IMG_DIR, transform=val_tf)
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
        )

        model = get_model()
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = OneCycleLR(
            optimizer, max_lr=3e-4, steps_per_epoch=len(train_loader), epochs=EPOCHS
        )

        for epoch in range(EPOCHS):
            _ = train_one_epoch(model, train_loader, criterion, optimizer, scheduler)
        auc = evaluate_with_tta(model, val_loader)
        print(f"Fold {fold+1} AUROC: {auc:.4f}")
        fold_aucs.append(auc)
        # Save this fold
        torch.save(
            model.state_dict(), os.path.join(WORKING_DIR, f"model_fold{fold}.pth")
        )

    mean_auc = np.mean(fold_aucs)
    print(f"Mean CV AUROC: {mean_auc:.4f}")

    # Also train final on full data if desired (kept for compatibility)
    freqs_full = df.skin_tone.value_counts().to_dict()
    w_full = df.skin_tone.map(lambda x: 1.0 / freqs_full[x]).values
    sampler_full = WeightedRandomSampler(w_full, len(w_full), replacement=True)
    full_ds = SkinLesionDataset(df, IMG_DIR, transform=train_tf)
    full_loader = DataLoader(
        full_ds, batch_size=BATCH_SIZE, sampler=sampler_full, num_workers=4
    )
    final_model = get_model()
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(final_model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer, max_lr=3e-4, steps_per_epoch=len(full_loader), epochs=EPOCHS
    )
    for _ in range(EPOCHS):
        _ = train_one_epoch(final_model, full_loader, criterion, optimizer, scheduler)
    torch.save(final_model.state_dict(), MODEL_PATH)
    print("Final full‐data model saved.")

    # Generate ensemble test predictions
    if os.path.exists(TEST_DIR) and os.listdir(TEST_DIR):
        preds = predict(WORKING_DIR, TEST_DIR)
        submission = pd.DataFrame.from_dict(
            preds, orient="index", columns=["malignancy_probability"]
        )
        submission.index.name = "image_name"
        submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"))
        print("Saved submission.csv with ensemble test predictions.")
