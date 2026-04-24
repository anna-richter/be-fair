import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


class ImagePathsDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, idx


def extract_features(image_paths, feat_model, device, transform, batch_size=32):
    ds = ImagePathsDataset(image_paths, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    feat_dim = feat_model.feature_dim
    all_feats = np.zeros((len(image_paths), feat_dim), dtype=np.float32)
    feat_model.eval()
    with torch.no_grad():
        for imgs, idxs in loader:
            imgs = imgs.to(device)
            feats = feat_model.forward_features(imgs)
            if hasattr(feat_model, "global_pool"):
                feats = feat_model.global_pool(feats)
            if feats.ndim > 2:
                feats = torch.flatten(feats, start_dim=1)
            np_feats = feats.cpu().numpy()
            all_feats[idxs.numpy()] = np_feats
    return all_feats


def predict(
    image_paths, feat_model, mlp_models, device="cpu", transforms=None, batch_size=32
):
    """
    Predict malignancy probabilities using TTA transforms list.
    Args:
        image_paths (List[str])
        feat_model (torch.nn.Module)
        mlp_models (List[sklearn Pipeline])
        device (str)
        transforms (List[callable]): list of preprocessing transforms (e.g. [orig, flip])
        batch_size (int)
    Returns:
        np.ndarray: averaged probabilities over transforms and models.
    """
    if transforms is None:
        raise ValueError("Please provide a list of transforms for TTA.")
    # for each transform, extract features
    tta_probs = []
    for transform in transforms:
        feats = extract_features(image_paths, feat_model, device, transform, batch_size)
        # predict with each model
        probs = np.stack([m.predict_proba(feats)[:, 1] for m in mlp_models], axis=1)
        tta_probs.append(probs.mean(axis=1))
    # average across transforms
    return np.mean(np.stack(tta_probs, axis=1), axis=1)


def main():
    IMG_DIR = "./input/MyImages"
    CSV_PATH = "./input/mydataset.csv"
    N_SPLITS = 5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # load labels
    df = pd.read_csv(CSV_PATH)
    df["label_bin"] = (df.label == "malignant").astype(int)

    # map image names
    image_paths = []
    for base in df.image_name:
        matches = glob.glob(os.path.join(IMG_DIR, f"*{base}.*"))
        if not matches:
            raise FileNotFoundError(f"No file found for image {base}")
        image_paths.append(matches[0])

    # transforms
    transform_orig = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    transform_flip = T.Compose(
        [
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # load backbone
    backbone = timm.create_model("efficientnet_b0", pretrained=True)
    backbone.eval()
    backbone.feature_dim = backbone.num_features
    backbone = backbone.to(DEVICE)

    # extract original & flipped features
    print("Extracting original features...")
    features_orig = extract_features(
        image_paths, backbone, DEVICE, transform_orig, batch_size=32
    )
    print("Extracting flipped features...")
    features_flip = extract_features(
        image_paths, backbone, DEVICE, transform_flip, batch_size=32
    )

    labels = df.label_bin.values
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    mlp_models = []
    scores = []

    # 5-fold CV: train on orig features, eval avg of orig+flip
    for fold, (tr_idx, va_idx) in enumerate(skf.split(features_orig, labels), 1):
        X_tr, y_tr = features_orig[tr_idx], labels[tr_idx]
        X_va_o, X_va_f, y_va = (
            features_orig[va_idx],
            features_flip[va_idx],
            labels[va_idx],
        )
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(128,),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        early_stopping=True,
                        random_state=42,
                        max_iter=200,
                    ),
                ),
            ]
        )
        pipe.fit(X_tr, y_tr)
        # preds orig & flip
        p_o = pipe.predict_proba(X_va_o)[:, 1]
        p_f = pipe.predict_proba(X_va_f)[:, 1]
        p_avg = 0.5 * (p_o + p_f)
        auc = roc_auc_score(y_va, p_avg)
        print(f"Fold {fold} AUROC w/ TTA: {auc:.4f}")
        scores.append(auc)
        mlp_models.append(pipe)

    print(f"Mean AUROC w/ TTA: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # final predictions with TTA
    preds = predict(
        image_paths,
        backbone,
        mlp_models,
        device=DEVICE,
        transforms=[transform_orig, transform_flip],
        batch_size=32,
    )
    submission = pd.DataFrame({"image_name": df.image_name, "malignant_prob": preds})
    os.makedirs("./working", exist_ok=True)
    submission.to_csv("./working/submission.csv", index=False)
    print("Saved submission.csv to ./working/")


if __name__ == "__main__":
    main()
