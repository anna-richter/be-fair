import os
import joblib
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
import timm

# Configuration
# start change
# DATA_CSV = "./input/mydataset.csv"  # original
DATA_CSV = "/home/anri21/be-fair/aide/MyData/mydataset.csv"
# end change
# start change
# IMAGE_DIR = "./input/MyImages"  # original
IMAGE_DIR = "/home/anri21/be-fair/aide/MyData/MyImages"
# end change
MODEL_PATH = "./working/lgbm_model.pkl"
SUBMISSION_PATH = "./working/submission.csv"
# start change
os.makedirs("./working", exist_ok=True)
# end change
BATCH_SIZE = 32
N_FOLDS = 5
SEED = 42


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), os.path.basename(path)


def extract_features(model, device, image_paths, transform):
    model.eval()
    ds = ImageDataset(image_paths, transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    feats_list, names = [], []
    with torch.no_grad():
        for imgs, batch_names in loader:
            imgs = imgs.to(device)
            feats = model.forward_features(imgs)
            if feats.ndim == 3:
                feats = feats[:, 0]
            feats_list.append(feats.cpu().numpy())
            names.extend(batch_names)
    return np.vstack(feats_list), names


def extract_features_tta(model, device, image_paths):
    base = [transforms.Resize(256), transforms.CenterCrop(224)]
    norm = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    flips = [False, True]
    angles = [0, 90, 180, 270]
    all_feats = None
    names_ref = None
    for flip in flips:
        for angle in angles:
            ops = []
            if flip:
                ops.append(transforms.Lambda(lambda img: F.hflip(img)))
            if angle != 0:
                ops.append(transforms.Lambda(lambda img, a=angle: F.rotate(img, a)))
            tfm = transforms.Compose(base + ops + norm)
            feats, names = extract_features(model, device, image_paths, tfm)
            if all_feats is None:
                all_feats = feats
                names_ref = names
            else:
                all_feats += feats
    all_feats /= len(flips) * len(angles)
    return all_feats, names_ref


def predict(image_folder: str) -> pd.DataFrame:
    files = sorted(
        [
            f
            for f in os.listdir(image_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    paths = [os.path.join(image_folder, f) for f in files]
    lgbm = joblib.load(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit = timm.create_model("vit_base_patch16_224", pretrained=True)
    vit.reset_classifier(0)
    vit.to(device)
    feats, names = extract_features_tta(vit, device, paths)
    probs = lgbm.predict(feats, num_iteration=lgbm.best_iteration)
    return pd.DataFrame(
        {
            "image_name": [os.path.splitext(n)[0] for n in names],
            "malignancy_probability": probs,
        }
    )


if __name__ == "__main__":
    df = pd.read_csv(DATA_CSV)
    df["filepath"] = df["image_name"].apply(
        lambda x: os.path.join(IMAGE_DIR, f"{x}.jpg")
    )
    df["target"] = (df["label"] == "malignant").astype(int)
    paths = df["filepath"].tolist()
    y = df["target"].values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit = timm.create_model("vit_base_patch16_224", pretrained=True)
    vit.reset_classifier(0)
    vit.to(device)

    print("Extracting TTA features (8-way: rotations × flips)...")
    X, _ = extract_features_tta(vit, device, paths)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    aucs, best_iters = [], []
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "seed": SEED,
    }
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_val],
            num_boost_round=1000,
            callbacks=[
                lgb.callback.early_stopping(stopping_rounds=50),
                lgb.callback.log_evaluation(period=0),
            ],
        )
        preds = model.predict(X_val, num_iteration=model.best_iteration)
        auc = roc_auc_score(y_val, preds)
        aucs.append(auc)
        best_iters.append(model.best_iteration)
        print(f"Fold {fold} AUC: {auc:.4f}, best_iter={model.best_iteration}")
    mean_auc = np.mean(aucs)
    print(f"Mean CV AUC: {mean_auc:.4f}")

    avg_iter = int(np.mean(best_iters) * 1.1)
    print(f"Retraining on full data for {avg_iter} rounds...")
    lgb_full = lgb.Dataset(X, label=y)
    final_model = lgb.train(params, lgb_full, num_boost_round=avg_iter)
    joblib.dump(final_model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # start change
    # preds_df = predict(IMAGE_DIR)  # original
    # preds_df.to_csv(SUBMISSION_PATH, index=False)  # original
    _ddi_files = sorted(os.listdir("/home/anri21/be-fair/evaluation/DDI/images"))
    _sub = predict("/home/anri21/be-fair/evaluation/DDI/images")
    _pmap = dict(zip(_sub.iloc[:, 0], _sub.iloc[:, 1].astype(float)))
    pd.DataFrame({
        "DDI_file": _ddi_files,
        "predicted_probability": [_pmap[os.path.splitext(f)[0]] for f in _ddi_files]
    }).to_csv("./DDI_predictions.csv", index=False)
    # end change
    print(f"Saved predictions to {SUBMISSION_PATH}")
