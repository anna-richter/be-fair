import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import OneCycleLR

# Seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class SkinLesionDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
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
        label = torch.tensor(row.label, dtype=torch.float32)
        weight = torch.tensor(row.sample_weight, dtype=torch.float32)
        return img, label, weight


def mixup_data(x, y, w, alpha=0.2):
    """Returns mixed inputs, pairs of targets, and mixed weights"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    w_a, w_b = w, w[index]
    mixed_w = lam * w_a + (1 - lam) * w_b
    return mixed_x, y_a, y_b, mixed_w, lam


# Load and prepare data
# start change
# df = pd.read_csv("./input/mydataset.csv")  # original
df = pd.read_csv("/home/anri21/be-fair/aide/MyData/mydataset.csv")
# end change
df["label"] = (df.label == "malignant").astype(int)
tone_counts = df.skin_tone.value_counts().to_dict()
inv_freq = {k: 1.0 / v for k, v in tone_counts.items()}
mean_inv = np.mean(list(inv_freq.values()))
inv_freq = {k: v / mean_inv for k, v in inv_freq.items()}
df["sample_weight"] = df.skin_tone.map(inv_freq).fillna(1.0)

# Transforms
train_tf = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
val_tf = train_tf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TTA components
tta_resize = transforms.Resize(256)
tta_tencrop = transforms.TenCrop(224)
tta_to_tensor = transforms.ToTensor()
tta_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)


def tta_inference(model, img):
    img_resized = tta_resize(img)
    crops = tta_tencrop(img_resized)
    batch = torch.stack([tta_normalize(tta_to_tensor(c)) for c in crops]).to(device)
    with torch.no_grad():
        logits = model(batch).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs.mean()


# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
aucs = []
epochs = 5

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df.label)):
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    # start change
    # train_ds = SkinLesionDataset(train_df, "./input/MyImages", transform=train_tf)  # original
    train_ds = SkinLesionDataset(train_df, "/home/anri21/be-fair/aide/MyData/MyImages", transform=train_tf)
    # end change
    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )

    # Model
    model = models.densenet121(pretrained=True)
    for name, param in model.features.named_parameters():
        if not name.startswith("denseblock4") and not name.startswith("norm5"):
            param.requires_grad = False
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    model = model.to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    scheduler = OneCycleLR(
        optimizer, max_lr=1e-3, epochs=epochs, steps_per_epoch=len(train_loader)
    )
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    # Training with Mixup
    model.train()
    for epoch in range(epochs):
        for imgs, labels, weights in train_loader:
            imgs, labels, weights = (
                imgs.to(device),
                labels.to(device),
                weights.to(device),
            )
            mixed_x, y_a, y_b, mixed_w, lam = mixup_data(
                imgs, labels, weights, alpha=0.2
            )
            logits = model(mixed_x).squeeze(1)
            loss_a = criterion(logits, y_a)
            loss_b = criterion(logits, y_b)
            loss = lam * loss_a + (1 - lam) * loss_b
            loss = (loss * mixed_w).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    # Validation with TTA
    model.eval()
    preds, truths = [], []
    for _, row in val_df.iterrows():
        # start change
        # img = Image.open(os.path.join("./input/MyImages", row.image_name + ".jpg")).convert("RGB")  # original
        img = Image.open(
            os.path.join("/home/anri21/be-fair/aide/MyData/MyImages", row.image_name + ".jpg")
        ).convert("RGB")
        # end change
        prob = tta_inference(model, img)
        preds.append(prob)
        truths.append(row.label)
    auc = roc_auc_score(truths, preds)
    print(f"Fold {fold} AUROC with Mixup+TTA: {auc:.4f}")
    aucs.append(auc)

mean_auc, std_auc = np.mean(aucs), np.std(aucs)
print(f"Mean CV AUROC with Mixup+TTA: {mean_auc:.4f} ± {std_auc:.4f}")

# Retrain on full data
# start change
# full_ds = SkinLesionDataset(df, "./input/MyImages", transform=train_tf)  # original
full_ds = SkinLesionDataset(df, "/home/anri21/be-fair/aide/MyData/MyImages", transform=train_tf)
# end change
full_loader = DataLoader(
    full_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
)
model = models.densenet121(pretrained=True)
for name, param in model.features.named_parameters():
    if not name.startswith("denseblock4") and not name.startswith("norm5"):
        param.requires_grad = False
model.classifier = nn.Linear(model.classifier.in_features, 1)
model = model.to(device)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
scheduler = OneCycleLR(
    optimizer, max_lr=1e-3, epochs=epochs, steps_per_epoch=len(full_loader)
)
criterion = nn.BCEWithLogitsLoss(reduction="none")

model.train()
for epoch in range(epochs):
    for imgs, labels, weights in full_loader:
        imgs, labels, weights = imgs.to(device), labels.to(device), weights.to(device)
        mixed_x, y_a, y_b, mixed_w, lam = mixup_data(imgs, labels, weights, alpha=0.2)
        logits = model(mixed_x).squeeze(1)
        loss_a = criterion(logits, y_a)
        loss_b = criterion(logits, y_b)
        loss = lam * loss_a + (1 - lam) * loss_b
        loss = (loss * mixed_w).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

os.makedirs("./working", exist_ok=True)
torch.save(model.state_dict(), "./working/final_model.pth")


def predict(folder: str):
    """
    Predict malignancy probability for each .jpg image in the given folder using TenCrop TTA.
    Args:
      folder (str): Path to directory of .jpg images.
    Returns:
      Dict[str, float]: Mapping from image filename to malignancy probability.
    """
    mdl = models.densenet121(pretrained=False)
    mdl.classifier = nn.Linear(mdl.classifier.in_features, 1)
    mdl.load_state_dict(torch.load("./working/final_model.pth", map_location=device))
    mdl = mdl.to(device).eval()
    results = {}
    for fname in os.listdir(folder):
        # start change
        # if not fname.lower().endswith(".jpg"):  # original
        if not fname.lower().endswith((".jpg", ".png")):
        # end change
            continue
        img = Image.open(os.path.join(folder, fname)).convert("RGB")
        results[fname] = tta_inference(mdl, img)
    return results


# start change
_ddi_files = sorted(os.listdir("/home/anri21/be-fair/evaluation/DDI/images"))
_preds = predict("/home/anri21/be-fair/evaluation/DDI/images")
pd.DataFrame({
    "DDI_file": _ddi_files,
    "predicted_probability": [float(_preds[f]) for f in _ddi_files]
}).to_csv("./DDI_predictions.csv", index=False)
# end change
