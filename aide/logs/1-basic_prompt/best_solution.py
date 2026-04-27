import os
import random
import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

# Configuration
DATA_CSV = "./input/mydataset.csv"
IMAGES_DIR = "./input/MyImages"
MODEL_DIR = "./working"
os.makedirs(MODEL_DIR, exist_ok=True)
BATCH_SIZE = 32
NUM_EPOCHS = 5
LR = 1e-4
WEIGHT_DECAY = 1e-2  # Added weight decay for AdamW
IMAGE_SIZE = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
CUTMIX_PROB = 0.5
CUTMIX_ALPHA = 1.0
NUM_FOLDS = 5

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.benchmark = True


class SkinDataset(Dataset):
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
        label = torch.tensor(row["binary_label"], dtype=torch.float32)
        return image, label


def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value="random"
        ),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
val_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def train_and_evaluate():
    df = pd.read_csv(DATA_CSV)
    df["binary_label"] = df["label"].apply(lambda x: 1 if x == "malignant" else 0)
    X = df["image_name"].values
    y = df["binary_label"].values
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    fold_aurocs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold+1}/{NUM_FOLDS}")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        # Weighted sampler for class balance
        class_counts = train_df["binary_label"].value_counts().to_dict()
        weights = train_df["binary_label"].map(lambda x: 1.0 / class_counts[x]).values
        sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )

        train_ds = SkinDataset(train_df, IMAGES_DIR, transform=train_transform)
        val_ds = SkinDataset(val_df, IMAGES_DIR, transform=val_transform)
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
        )

        # EfficientNet-B3 backbone
        model = timm.create_model("efficientnet_b3", pretrained=True)
        in_f = model.classifier.in_features
        model.classifier = nn.Linear(in_f, 1)
        model = model.to(DEVICE)

        criterion = nn.BCEWithLogitsLoss(reduction="none")
        # Use AdamW with weight decay for regularization
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )

        best_auc = 0.0
        best_path = os.path.join(MODEL_DIR, f"effb3_fold{fold}.pth")
        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
                optimizer.zero_grad()
                if random.random() < CUTMIX_PROB:
                    lam = np.random.beta(CUTMIX_ALPHA, CUTMIX_ALPHA)
                    idx2 = torch.randperm(imgs.size(0)).to(DEVICE)
                    ta, tb = labels, labels[idx2]
                    x1, y1, x2, y2 = rand_bbox(imgs.size(), lam)
                    imgs[:, :, y1:y2, x1:x2] = imgs[idx2, :, y1:y2, x1:x2]
                    lam_adj = 1 - ((x2 - x1) * (y2 - y1) / (IMAGE_SIZE * IMAGE_SIZE))
                    outputs = model(imgs)
                    loss = (
                        lam_adj * criterion(outputs, ta).mean()
                        + (1 - lam_adj) * criterion(outputs, tb).mean()
                    )
                else:
                    outputs = model(imgs)
                    loss = criterion(outputs, labels).mean()
                loss.backward()
                optimizer.step()

            model.eval()
            preds, truths = [], []
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs = imgs.to(DEVICE)
                    o1 = model(imgs)
                    p1 = torch.sigmoid(o1)
                    o2 = model(torch.flip(imgs, [3]))
                    p2 = torch.sigmoid(o2)
                    p = ((p1 + p2) / 2).cpu().numpy().flatten()
                    preds.extend(p)
                    truths.extend(labels.numpy().flatten())
            auc = roc_auc_score(truths, preds)
            print(f" Epoch {epoch}: Val AUROC {auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), best_path)
        print(f" Fold {fold+1} best AUROC: {best_auc:.4f}\n")
        fold_aurocs.append(best_auc)

    mean_auc = np.mean(fold_aurocs)
    print("All fold AUROCs:", fold_aurocs)
    print(f"Mean CV AUROC: {mean_auc:.4f}")


def predict(image_folder, model_folder=MODEL_DIR, batch_size=32):
    """
    Predict malignancy probabilities using ensemble of EfficientNet-B3 fold models with TTA.
    Args:
      image_folder (str): path to folder of images (.jpg).
      model_folder (str): folder containing fold model .pth files.
      batch_size (int): inference batch size.
    Returns:
      dict: filename -> malignancy probability.
    """
    model_paths = sorted(
        [
            os.path.join(model_folder, f)
            for f in os.listdir(model_folder)
            if f.startswith("effb3_fold") and f.endswith(".pth")
        ]
    )
    models_ens = []
    for pth in model_paths:
        m = timm.create_model("efficientnet_b3", pretrained=False)
        in_f = m.classifier.in_features
        m.classifier = nn.Linear(in_f, 1)
        m.load_state_dict(torch.load(pth, map_location=DEVICE))
        m = m.to(DEVICE).eval()
        models_ens.append(m)

    img_files = [f for f in os.listdir(image_folder) if f.lower().endswith(".jpg")]
    results = {}
    for i in range(0, len(img_files), batch_size):
        batch = img_files[i : i + batch_size]
        imgs = []
        for fn in batch:
            img = Image.open(os.path.join(image_folder, fn)).convert("RGB")
            imgs.append(val_transform(img))
        imgs = torch.stack(imgs).to(DEVICE)
        with torch.no_grad():
            sum_p = torch.zeros(len(batch), device=DEVICE)
            for m in models_ens:
                o1 = m(imgs)
                p1 = torch.sigmoid(o1).flatten()
                o2 = m(torch.flip(imgs, [3]))
                p2 = torch.sigmoid(o2).flatten()
                sum_p += (p1 + p2) / 2
            avg = (sum_p / len(models_ens)).cpu().numpy()
        for fn, pr in zip(batch, avg):
            results[fn] = float(pr)
    return results


if __name__ == "__main__":
    train_and_evaluate()
    # Example usage: preds = predict("./input/NewImages"); print(preds)
