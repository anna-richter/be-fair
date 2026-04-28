import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR

# Configuration
# start change
# DATA_CSV = "input/mydataset.csv"  # original
DATA_CSV = "/home/anri21/be-fair/aide/MyData/mydataset.csv"
# end change
# start change
# IMG_DIR = "input/MyImages"  # original
IMG_DIR = "/home/anri21/be-fair/aide/MyData/MyImages"
# end change
MODEL_PATH = "working/model.pth"
SUBMISSION_PATH = "working/submission.csv"
N_SPLITS = 5
BATCH_SIZE = 32
N_EPOCHS = 5
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("working", exist_ok=True)
torch.manual_seed(42)
np.random.seed(42)


# Dataset definition
class SkinDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.names = df["image_name"].tolist()
        self.labels = df["label_bin"].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name = self.names[idx]
        img = Image.open(os.path.join(self.img_dir, name + ".jpg")).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label


# Transforms
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Load data
df = pd.read_csv(DATA_CSV)
df["label_bin"] = (df["label"] == "malignant").astype(int)
y = df["label_bin"].values


# TTA function
def tta_predictions(model, imgs):
    views = [None, (3,), (2,), (2, 3)]
    logits_sum = torch.zeros(imgs.size(0), 1, device=imgs.device)
    for d in views:
        inp = imgs if d is None else torch.flip(imgs, dims=d)
        logits_sum += model(inp)
    return logits_sum / len(views)


# Cross-validation
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
aucs = []
for fold, (train_idx, val_idx) in enumerate(skf.split(df, y), 1):
    print(f"Fold {fold}")
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    train_ds = SkinDataset(train_df, IMG_DIR, train_transform)
    val_ds = SkinDataset(val_df, IMG_DIR, val_transform)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    model = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)

    for epoch in range(N_EPOCHS):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            preds = model(imgs)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            logits = tta_predictions(model, imgs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy().flatten())
    auc = roc_auc_score(all_labels, all_probs)
    print(f"  Fold {fold} AUROC (4-view TTA): {auc:.4f}")
    aucs.append(auc)

mean_auc, std_auc = np.mean(aucs), np.std(aucs)
print(f"\nCV AUROC with CosineAnnealingLR & 4-view TTA: {mean_auc:.4f} ± {std_auc:.4f}")

# Retrain on full data
full_ds = SkinDataset(df, IMG_DIR, train_transform)
full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
model = models.densenet121(pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, 1)
model = model.to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)

model.train()
for epoch in range(N_EPOCHS):
    for imgs, labels in tqdm(full_loader, desc=f"Full train epoch {epoch+1}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
        preds = model(imgs)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
torch.save(model.state_dict(), MODEL_PATH)


# Prediction function
def predict(folder: str, model_path: str = MODEL_PATH, device=DEVICE):
    """
    Predict malignancy probabilities for images in a folder using 4-view TTA.
    Returns a DataFrame with columns ['image_name', 'malignancy_probability'].
    """
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()
    transform = val_transform
    results = []
    # start change
    # img_names = sorted([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])  # original
    img_names = sorted([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))])
    # end change
    with torch.no_grad():
        for i in range(0, len(img_names), BATCH_SIZE):
            batch = img_names[i : i + BATCH_SIZE]
            imgs = torch.stack(
                [
                    transform(Image.open(os.path.join(folder, f)).convert("RGB"))
                    for f in batch
                ]
            ).to(device)
            logits = tta_predictions(model, imgs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            for fname, p in zip(batch, probs):
                results.append((os.path.splitext(fname)[0], float(p)))
    return pd.DataFrame(results, columns=["image_name", "malignancy_probability"])


# start change
# sub = predict(IMG_DIR)  # original
# sub.to_csv(SUBMISSION_PATH, index=False)  # original
_ddi_files = sorted(os.listdir("/home/anri21/be-fair/evaluation/DDI/images"))
_sub = predict("/home/anri21/be-fair/evaluation/DDI/images")
_pmap = dict(zip(_sub.iloc[:, 0], _sub.iloc[:, 1].astype(float)))
pd.DataFrame({
    "DDI_file": _ddi_files,
    "predicted_probability": [_pmap[os.path.splitext(f)[0]] for f in _ddi_files]
}).to_csv("./DDI_predictions.csv", index=False)
# end change
print(f"Submission saved to {SUBMISSION_PATH}")
