import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


class SkinDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), idx


def extract_embeddings(model, dataloader, device, feature_dim):
    model.eval()
    emb = np.zeros((len(dataloader.dataset), feature_dim), dtype=np.float32)
    with torch.no_grad():
        for imgs, idxs in dataloader:
            imgs = imgs.to(device)
            feats = model(imgs)
            emb[idxs.numpy()] = feats.cpu().numpy()
    return emb


def predict(image_paths, classifiers, batch_size=32, num_workers=4):
    """
    Predict malignancy probability for a list of image paths with TTA including 90° rotation.
    Args:
      image_paths: list of str paths to images
      classifiers: list of trained sklearn classifiers with predict_proba
    Returns:
      numpy array of shape (n_images,) with average malignant probability
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    base_transform = weights.transforms()
    hflip = transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), base_transform])
    vflip = transforms.Compose([transforms.RandomVerticalFlip(p=1.0), base_transform])
    rot90 = transforms.Compose(
        [transforms.Lambda(lambda img: img.rotate(90)), base_transform]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = efficientnet_b0(weights=weights)
    feature_dim = model.classifier[1].in_features
    model.classifier = torch.nn.Identity()
    model.to(device)

    # original
    ds0 = SkinDataset(image_paths, base_transform)
    dl0 = DataLoader(ds0, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    emb0 = extract_embeddings(model, dl0, device, feature_dim)
    # horizontal flip
    ds1 = SkinDataset(image_paths, hflip)
    dl1 = DataLoader(ds1, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    emb1 = extract_embeddings(model, dl1, device, feature_dim)
    # vertical flip
    ds2 = SkinDataset(image_paths, vflip)
    dl2 = DataLoader(ds2, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    emb2 = extract_embeddings(model, dl2, device, feature_dim)
    # 90° rotation
    ds3 = SkinDataset(image_paths, rot90)
    dl3 = DataLoader(ds3, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    emb3 = extract_embeddings(model, dl3, device, feature_dim)

    emb = (emb0 + emb1 + emb2 + emb3) / 4.0

    preds = np.zeros(len(image_paths), dtype=np.float32)
    for clf in classifiers:
        preds += clf.predict_proba(emb)[:, 1]
    preds /= len(classifiers)
    return preds


if __name__ == "__main__":
    # load data
    df = pd.read_csv("./input/mydataset.csv")
    df["y"] = (df["label"] == "malignant").astype(int)
    img_dir = "./input/MyImages"
    paths = df["image_name"].apply(lambda x: os.path.join(img_dir, x + ".jpg")).tolist()
    ys = df["y"].values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    base_transform = weights.transforms()
    hflip = transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), base_transform])
    vflip = transforms.Compose([transforms.RandomVerticalFlip(p=1.0), base_transform])
    rot90 = transforms.Compose(
        [transforms.Lambda(lambda img: img.rotate(90)), base_transform]
    )

    # load backbone
    model = efficientnet_b0(weights=weights)
    feature_dim = model.classifier[1].in_features
    model.classifier = torch.nn.Identity()
    model.to(device)

    # extract embeddings with TTA: orig, H-flip, V-flip, 90° rot
    ds0 = SkinDataset(paths, base_transform)
    dl0 = DataLoader(ds0, batch_size=32, num_workers=4, shuffle=False)
    emb0 = extract_embeddings(model, dl0, device, feature_dim)
    ds1 = SkinDataset(paths, hflip)
    dl1 = DataLoader(ds1, batch_size=32, num_workers=4, shuffle=False)
    emb1 = extract_embeddings(model, dl1, device, feature_dim)
    ds2 = SkinDataset(paths, vflip)
    dl2 = DataLoader(ds2, batch_size=32, num_workers=4, shuffle=False)
    emb2 = extract_embeddings(model, dl2, device, feature_dim)
    ds3 = SkinDataset(paths, rot90)
    dl3 = DataLoader(ds3, batch_size=32, num_workers=4, shuffle=False)
    emb3 = extract_embeddings(model, dl3, device, feature_dim)
    embeddings = (emb0 + emb1 + emb2 + emb3) / 4.0

    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    classifiers, aucs = [], []
    for fold, (tr, val) in enumerate(skf.split(embeddings, ys), 1):
        X_tr, X_val = embeddings[tr], embeddings[val]
        y_tr, y_val = ys[tr], ys[val]
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_tr, y_tr)
        p = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, p)
        print(f"Fold {fold} AUROC: {auc:.4f}")
        classifiers.append(clf)
        aucs.append(auc)
    print(f"Mean AUROC: {np.mean(aucs):.4f}")

    # overall on training via predict()
    preds = predict(paths, classifiers)
    print(f"Overall AUROC by predict(): {roc_auc_score(ys, preds):.4f}")
