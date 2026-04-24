import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import timm
from torchvision import transforms

# Paths
CSV_PATH = "./input/mydataset.csv"
IMG_DIR = "./input/MyImages"
SUBMISSION_PATH = "./working/submission.csv"

# Load metadata
df = pd.read_csv(CSV_PATH)
df["target"] = (df["label"] == "malignant").astype(int)

# Build mapping from image_name to file name
file_map = {}
for fname in os.listdir(IMG_DIR):
    if fname.startswith("."):
        continue
    key = os.path.splitext(fname)[0]
    file_map[key] = fname

# Transforms (no augmentation)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)


class SkinDataset(Dataset):
    def __init__(self, df, img_dir, file_map, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.file_map = file_map
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        key = row["image_name"]
        fname = self.file_map[key]
        path = os.path.join(self.img_dir, fname)
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        y = row["target"]
        return x, y


def extract_features(model, loader, device):
    model.eval()
    feats = []
    labels = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            f = model(xb).view(xb.size(0), -1).cpu().numpy()
            feats.append(f)
            labels.extend(yb.numpy().tolist())
    X = np.vstack(feats)
    y = np.array(labels)
    return X, y


def predict(feature_extractor, classifier, image_paths, transform, device=None):
    """
    Predict malignancy probabilities for a list of image file paths.
    Args:
        feature_extractor: frozen torch model ending with global pool (no fc)
        classifier: sklearn classifier with predict_proba
        image_paths: list of full image paths
        transform: torchvision transform
        device: torch device or None
    Returns:
        List of (image_name, probability) tuples.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = feature_extractor.to(device).eval()
    results = []
    with torch.no_grad():
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            f = feature_extractor(x).view(1, -1).cpu().numpy()
            p = classifier.predict_proba(f)[0, 1]
            name = os.path.splitext(os.path.basename(path))[0]
            results.append((name, p))
    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Build dataset & dataloader
    ds = SkinDataset(df, IMG_DIR, file_map, transform)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    # 2) Load EfficientNet-B0 and remove the final classifier
    full_model = timm.create_model("efficientnet_b0", pretrained=True)
    # children up to global_pool
    backbone = nn.Sequential(*list(full_model.children())[:-1]).to(device)

    # 3) Extract features
    X, y = extract_features(backbone, loader, device)

    # 4) 5-fold CV with LogisticRegression
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        clf = LogisticRegression(
            max_iter=1000, class_weight="balanced", solver="liblinear"
        )
        clf.fit(X_tr, y_tr)
        preds = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        print(f"Fold {fold} AUC: {auc:.4f}")
        aucs.append(auc)
    mean_auc = np.mean(aucs)
    print(f"Mean AUC: {mean_auc:.4f}")

    # 5) Retrain on full data
    final_clf = LogisticRegression(
        max_iter=1000, class_weight="balanced", solver="liblinear"
    )
    final_clf.fit(X, y)

    # 6) Predict on all images and save submission
    all_paths = [
        os.path.join(IMG_DIR, fn)
        for fn in os.listdir(IMG_DIR)
        if not fn.startswith(".")
    ]
    preds = predict(backbone, final_clf, all_paths, transform, device)
    sub_df = pd.DataFrame(preds, columns=["image_name", "probability"])
    sub_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved submission to {SUBMISSION_PATH}")
