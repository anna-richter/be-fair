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

# Map image names to filenames (ignore hidden/temp files)
file_map = {
    os.path.splitext(f)[0]: f for f in os.listdir(IMG_DIR) if not f.startswith(".")
}

# Transforms
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
        fname = self.file_map[row["image_name"]]
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        return self.transform(img), row["target"]


class TestDataset(Dataset):
    def __init__(self, image_names, img_dir, file_map, transform):
        self.image_names = image_names
        self.img_dir = img_dir
        self.file_map = file_map
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        fname = self.file_map[name]
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        return self.transform(img), name


def extract_features(model, loader, device):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            f = model(xb).view(xb.size(0), -1).cpu().numpy()
            feats.append(f)
            labels.extend(yb.numpy().tolist())
    return np.vstack(feats), np.array(labels)


def predict(
    feature_extractors,
    classifier,
    img_dir,
    file_map,
    transform,
    batch_size=32,
    device=None,
):
    """
    Predict malignancy probabilities for all images in img_dir using batch processing.
    Args:
        feature_extractors: list of torch models ending with global pool (frozen).
        classifier: sklearn classifier with predict_proba.
        img_dir: directory containing images.
        file_map: dict mapping image_name -> filename.
        transform: torchvision transform.
        batch_size: batch size for DataLoader.
        device: torch device.
    Returns:
        List of (image_name, probability) tuples.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for m in feature_extractors:
        m.to(device).eval()

    image_names = sorted(file_map.keys())
    ds = TestDataset(image_names, img_dir, file_map, transform)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    results = []
    with torch.no_grad():
        for xb, names in loader:
            xb = xb.to(device)
            # Extract features for each backbone
            feats = [
                m(xb).view(xb.size(0), -1).cpu().numpy() for m in feature_extractors
            ]
            combined = np.hstack(feats)
            probs = classifier.predict_proba(combined)[:, 1]
            results.extend(zip(names, probs))
    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare training dataset & loader
    ds = SkinDataset(df, IMG_DIR, file_map, transform)
    loader = DataLoader(
        ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    # Load backbones and strip heads
    eff0 = timm.create_model("efficientnet_b0", pretrained=True)
    eff0 = nn.Sequential(*list(eff0.children())[:-1]).to(device)
    res50 = timm.create_model("resnet50", pretrained=True)
    res50 = nn.Sequential(*list(res50.children())[:-1]).to(device)

    # Extract features for all images in one pass per model
    X1, y = extract_features(eff0, loader, device)
    X2, _ = extract_features(res50, loader, device)
    X = np.hstack([X1, X2])

    # 5-fold CV evaluation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for fold, (tr, val) in enumerate(skf.split(X, y), 1):
        clf = LogisticRegression(
            max_iter=1000, class_weight="balanced", solver="liblinear"
        )
        clf.fit(X[tr], y[tr])
        preds = clf.predict_proba(X[val])[:, 1]
        auc = roc_auc_score(y[val], preds)
        print(f"Fold {fold} AUC: {auc:.4f}")
        aucs.append(auc)
    print(f"Mean AUC: {np.mean(aucs):.4f}")

    # Retrain on full data
    final_clf = LogisticRegression(
        max_iter=1000, class_weight="balanced", solver="liblinear"
    )
    final_clf.fit(X, y)

    # Batch predict on all images and save submission
    preds = predict(
        [eff0, res50],
        final_clf,
        IMG_DIR,
        file_map,
        transform,
        batch_size=32,
        device=device,
    )
    sub_df = pd.DataFrame(preds, columns=["image_name", "probability"])
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    sub_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved submission to {SUBMISSION_PATH}")
