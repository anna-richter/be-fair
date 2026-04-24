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


# TTA transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def make_transform(op):
    ops = [transforms.Resize((224, 224)), op, transforms.ToTensor(), normalize]
    return transforms.Compose([x for x in ops if x])


ttas = [
    make_transform(None),
    make_transform(transforms.Lambda(lambda img: img.transpose(Image.FLIP_LEFT_RIGHT))),
    make_transform(transforms.Lambda(lambda img: img.transpose(Image.FLIP_TOP_BOTTOM))),
    make_transform(transforms.Lambda(lambda img: img.rotate(90))),
    make_transform(transforms.Lambda(lambda img: img.rotate(180))),
    make_transform(transforms.Lambda(lambda img: img.rotate(270))),
]

# Load backbone
backbone = models.efficientnet_b0(pretrained=True)
backbone.classifier = nn.Identity()
backbone.to(DEVICE).eval()


def extract_features(image_names, img_dir=IMG_DIR):
    """Extract features for a list of image_names using TTA and EfficientNet."""
    all_feats = []
    for transform in ttas:
        ds = LesionDataset(image_names, img_dir=img_dir, transform=transform)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
        feats = []
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(DEVICE)
                out = backbone(imgs)
                feats.append(out.cpu().numpy())
        all_feats.append(np.vstack(feats))
    # average over TTAs
    return np.mean(np.stack(all_feats, axis=0), axis=0)


def predict(image_dir, clfs):
    """
    Predict malignant probabilities for all .jpg images in image_dir
    by averaging predictions from a list of LightGBM classifiers.
    Returns a DataFrame with ['image_name','malignant_probability'].
    """
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")]
    img_names = [os.path.splitext(f)[0] for f in files]
    feats = extract_features(img_names, img_dir=image_dir)
    # average probabilities across classifiers
    probs = np.zeros(len(img_names), dtype=float)
    for clf in clfs:
        probs += clf.predict_proba(feats)[:, 1]
    probs /= len(clfs)
    return pd.DataFrame({"image_name": img_names, "malignant_probability": probs})


if __name__ == "__main__":
    # Load data
    df = pd.read_csv(CSV_PATH)
    df["malignant"] = (df["label"] == "malignant").astype(int)
    names = df["image_name"].tolist()
    labels = df["malignant"].values

    # Precompute features
    print("Extracting features from all training images...")
    features = extract_features(names)

    # 5-fold CV with LightGBM + early stopping
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    fold_clfs = []

    print("Starting 5-fold CV...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels), 1):
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        clf = lgb.LGBMClassifier(
            random_state=RANDOM_STATE,
            n_estimators=10000,
            learning_rate=0.01,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
        )
        clf.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0),
            ],
        )
        fold_clfs.append(clf)
        preds = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        aucs.append(auc)
        print(f" Fold {fold}: AUC = {auc:.4f}, best_iter = {clf.best_iteration_}")

    mean_auc, std_auc = np.mean(aucs), np.std(aucs)
    print(f"\nMean ROC AUC: {mean_auc:.4f} ± {std_auc:.4f}")

    # Generate submission if test data exists
    if os.path.isdir(TEST_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print("\nGenerating submission.csv with ensemble of fold models...")
        submission = predict(TEST_DIR, fold_clfs)
        submission.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)
        print(f"Saved submission to {os.path.join(OUTPUT_DIR, 'submission.csv')}")
