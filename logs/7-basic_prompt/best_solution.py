import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# 1. Load metadata and labels
df = pd.read_csv("input/mydataset.csv")
df["binary_label"] = (df["label"] == "malignant").astype(int)
image_paths = (
    df["image_name"]
    .apply(lambda n: os.path.join("input", "MyImages", f"{n}.jpg"))
    .tolist()
)
y = df["binary_label"].values

# 2. Prepare device, backbone and transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enet = models.efficientnet_b0(pretrained=True)
feature_extractor = (
    nn.Sequential(enet.features, enet.avgpool, nn.Flatten()).to(device).eval()
)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def extract_features(paths):
    """Extract deep features with 4‐way TTA (orig, hflip, vflip, hvflip)."""
    feats = []
    with torch.no_grad():
        for p in paths:
            img = Image.open(p).convert("RGB")
            variants = [
                img,
                img.transpose(Image.FLIP_LEFT_RIGHT),
                img.transpose(Image.FLIP_TOP_BOTTOM),
                img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM),
            ]
            agg = None
            for v in variants:
                x = transform(v).unsqueeze(0).to(device)
                f = feature_extractor(x).cpu().numpy().reshape(-1)
                agg = f if agg is None else agg + f
            feats.append(agg / len(variants))
    return np.vstack(feats)


# 3. Extract features
features = extract_features(image_paths)

# 4. 5‐fold CV with LightGBM
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs = []
for train_idx, val_idx in skf.split(features, y):
    X_train, X_val = features[train_idx], features[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    clf = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_val)[:, 1]
    aucs.append(roc_auc_score(y_val, probs))
mean_auc = np.mean(aucs)
print(f"Mean 5-fold AUROC with EfficientNet-B0 + 4-way TTA + LightGBM: {mean_auc:.4f}")

# 5. Train final LightGBM classifier on all data
final_clf = lgb.LGBMClassifier(
    n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1
)
final_clf.fit(features, y)


def predict(image_dir):
    """
    Predict malignancy probability for images in a directory using EfficientNet-B0 + 4-way TTA + LightGBM.
    Args:
        image_dir (str): path to folder containing .jpg images
    Returns:
        pandas.DataFrame: columns ['image_name','malignant_prob']
    """
    img_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")])
    probs = []
    with torch.no_grad():
        for fname in img_files:
            path = os.path.join(image_dir, fname)
            img = Image.open(path).convert("RGB")
            variants = [
                img,
                img.transpose(Image.FLIP_LEFT_RIGHT),
                img.transpose(Image.FLIP_TOP_BOTTOM),
                img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM),
            ]
            agg = None
            for v in variants:
                x = transform(v).unsqueeze(0).to(device)
                f = feature_extractor(x).cpu().numpy().reshape(-1)
                agg = f if agg is None else agg + f
            feat = (agg / len(variants)).reshape(1, -1)
            p = final_clf.predict_proba(feat)[0, 1]
            probs.append(p)
    return pd.DataFrame({"image_name": img_files, "malignant_prob": probs})


# 6. Generate submission
submission = predict("input/MyImages")
os.makedirs("working", exist_ok=True)
submission.to_csv("working/submission.csv", index=False)
print("Saved submission.csv with malignancy probabilities.")
