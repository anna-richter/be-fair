import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Configuration
IMG_DIR = "./input/MyImages"
CSV_PATH = "./input/mydataset.csv"
TEST_DIR = "./input/test"
OUTPUT_DIR = "./working"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
RANDOM_STATE = 42


class LesionDataset(Dataset):
    def __init__(self, image_names, labels=None, img_dir=IMG_DIR, transform=None):
        self.image_names = image_names
        self.labels = labels
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        path = os.path.join(self.img_dir, name + ".jpg")
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.labels is not None:
            return img, self.labels[idx]
        else:
            return img, name


# Define six TTA transforms: original, h-flip, v-flip, rotate90, rotate180, rotate270
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
base_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ]
)
flip_h_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.flip(2)),
        normalize,
    ]
)
flip_v_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.flip(1)),
        normalize,
    ]
)
rotate90_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: img.rotate(90, expand=False)),
        transforms.ToTensor(),
        normalize,
    ]
)
rotate180_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: img.rotate(180, expand=False)),
        transforms.ToTensor(),
        normalize,
    ]
)
rotate270_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: img.rotate(270, expand=False)),
        transforms.ToTensor(),
        normalize,
    ]
)
ttas = [
    base_transform,
    flip_h_transform,
    flip_v_transform,
    rotate90_transform,
    rotate180_transform,
    rotate270_transform,
]

# Load data
df = pd.read_csv(CSV_PATH)
df["malignant"] = (df["label"] == "malignant").astype(int)
names = df["image_name"].tolist()
labels = df["malignant"].values

# Load pretrained EfficientNet-B0 as feature extractor
backbone = models.efficientnet_b0(pretrained=True)
backbone.classifier = nn.Identity()
backbone.to(DEVICE).eval()


def extract_features(image_names):
    all_feats = []
    for transform in ttas:
        ds = LesionDataset(image_names, transform=transform)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
        feats = []
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(DEVICE)
                out = backbone(imgs)
                feats.append(out.cpu().numpy())
        all_feats.append(np.vstack(feats))
    return np.mean(np.stack(all_feats, axis=0), axis=0)


# Precompute features
features = extract_features(names)

# 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
aucs = []
for train_idx, val_idx in skf.split(features, labels):
    X_train, X_val = features[train_idx], features[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]
    clf = lgb.LGBMClassifier(random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_val)[:, 1]
    aucs.append(roc_auc_score(y_val, preds))
mean_auc, std_auc = np.mean(aucs), np.std(aucs)
print(f"Mean ROC AUC: {mean_auc:.4f} ± {std_auc:.4f}")

# Retrain on full data
final_clf = lgb.LGBMClassifier(random_state=RANDOM_STATE)
final_clf.fit(features, labels)


def predict(image_dir):
    """
    Predict malignant probabilities for all .jpg images in image_dir.
    Returns a DataFrame with columns ['image_name','malignant_probability'].
    """
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")]
    img_names = [os.path.splitext(f)[0] for f in files]
    feats = extract_features(img_names)
    probs = final_clf.predict_proba(feats)[:, 1]
    return pd.DataFrame({"image_name": img_names, "malignant_probability": probs})


# Generate submission if test data exists
if os.path.isdir(TEST_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    submission = predict(TEST_DIR)
    submission.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)
