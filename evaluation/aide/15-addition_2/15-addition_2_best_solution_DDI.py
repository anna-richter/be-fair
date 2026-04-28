import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Configuration
# start change
# IMG_DIR = "./input/MyImages"  # original
IMG_DIR = "/home/anri21/be-fair/aide/MyData/MyImages"
# end change
# start change
# CSV_PATH = "./input/mydataset.csv"  # original
CSV_PATH = "/home/anri21/be-fair/aide/MyData/mydataset.csv"
# end change
MODEL_PATH = "./working/lgbm_model_tta_rot90.pkl"
SUBMISSION_PATH = "./working/submission.csv"
BATCH_SIZE = 32
SEED = 42
NUM_WORKERS = 4


# Dataset for embedding extraction
class SkinLesionDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.img_dir, row.image_name + ".jpg")
        img = Image.open(path).convert("RGB")
        return self.transform(img), row.skin_tone, row.label


# Load data
df = pd.read_csv(CSV_PATH)
df["label"] = (df["label"] == "malignant").astype(int)

# Transforms
base_tf = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)
hflip_tf = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)
vflip_tf = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)
rot90_tf = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Lambda(lambda img: img.rotate(90)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

# Device and feature model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feat_model = timm.create_model("efficientnet_b0", pretrained=True)
if hasattr(feat_model, "fc"):
    feat_model.fc = torch.nn.Identity()
elif hasattr(feat_model, "classifier"):
    feat_model.classifier = torch.nn.Identity()
feat_model.to(device).eval()


def extract_embeddings(transform):
    ds = SkinLesionDataset(df, IMG_DIR, transform)
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    feats, tones, labels = [], [], []
    with torch.no_grad():
        for imgs, tns, lbs in loader:
            imgs = imgs.to(device)
            emb = feat_model(imgs).cpu().numpy()
            feats.append(emb)
            tones.append(tns.numpy())
            labels.append(lbs.numpy())
    return np.vstack(feats), np.concatenate(tones), np.concatenate(labels)


# Extract embeddings under 4 TTAs
print("Extracting embeddings: original...")
fe_o, tones, labels = extract_embeddings(base_tf)
print("  horizontal flip...")
fe_h, _, _ = extract_embeddings(hflip_tf)
print("  vertical flip...")
fe_v, _, _ = extract_embeddings(vflip_tf)
print("  rotation 90°...")
fe_r, _, _ = extract_embeddings(rot90_tf)

# Compute averaged embeddings
fe_avg = (fe_o + fe_h + fe_v + fe_r) / 4.0

# 5-fold CV on averaged embeddings
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
aucs, gaps = [], []
for tr_idx, val_idx in skf.split(fe_avg, labels):
    Xtr, Xval = fe_avg[tr_idx], fe_avg[val_idx]
    ytr, yval = labels[tr_idx], labels[val_idx]
    tones_val = tones[val_idx]
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=SEED)
    clf.fit(Xtr, ytr)
    preds = clf.predict_proba(Xval)[:, 1]
    auc = roc_auc_score(yval, preds)
    aucs.append(auc)
    m_light = (tones_val > 0) & (tones_val <= 3)
    m_dark = tones_val >= 5
    if m_light.sum() > 0 and m_dark.sum() > 0:
        auc_l = roc_auc_score(yval[m_light], preds[m_light])
        auc_d = roc_auc_score(yval[m_dark], preds[m_dark])
        gaps.append(abs(auc_l - auc_d))
    else:
        gaps.append(0.0)

print(f"CV AUROC with 4-way TTA and train-on-avg: {np.mean(aucs):.4f}")
print(f"Avg AUROC gap (light vs dark): {np.mean(gaps):.4f}")

# Retrain on full averaged embeddings
final_clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=SEED)
final_clf.fit(fe_avg, labels)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(final_clf, f)


def predict(folder_path):
    """
    Predict malignancy probability using 4-way TTA (orig, hflip, vflip, rot90).
    Args:
      folder_path (str): Directory containing .jpg images.
    Returns:
      pd.DataFrame with columns ['image_name','malignancy_probability'].
    """
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    # start change
    # imgs = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]  # original
    imgs = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png"))]
    # end change
    results = []
    for i in range(0, len(imgs), BATCH_SIZE):
        batch = imgs[i : i + BATCH_SIZE]
        t_o, t_h, t_v, t_r = [], [], [], []
        for fn in batch:
            img = Image.open(os.path.join(folder_path, fn)).convert("RGB")
            t_o.append(base_tf(img))
            t_h.append(hflip_tf(img))
            t_v.append(vflip_tf(img))
            t_r.append(rot90_tf(img))
        bo = torch.stack(t_o).to(device)
        bh = torch.stack(t_h).to(device)
        bv = torch.stack(t_v).to(device)
        br = torch.stack(t_r).to(device)
        with torch.no_grad():
            e_o = feat_model(bo).cpu().numpy()
            e_h = feat_model(bh).cpu().numpy()
            e_v = feat_model(bv).cpu().numpy()
            e_r = feat_model(br).cpu().numpy()
        E = (e_o + e_h + e_v + e_r) / 4.0
        probs = model.predict_proba(E)[:, 1]
        results.extend(zip(batch, probs))
    return pd.DataFrame(results, columns=["image_name", "malignancy_probability"])


# start change
# test_folder = "./input/test_images"  # original
# if os.path.isdir(test_folder):       # original guard
#     df_sub = predict(test_folder)    # original
#     df_sub.to_csv(SUBMISSION_PATH, index=False)  # original
_ddi_files = sorted(os.listdir("/home/anri21/be-fair/evaluation/DDI/images"))
_sub = predict("/home/anri21/be-fair/evaluation/DDI/images")
_pmap = dict(zip(_sub.iloc[:, 0], _sub.iloc[:, 1].astype(float)))
pd.DataFrame({
    "DDI_file": _ddi_files,
    "predicted_probability": [_pmap[f] for f in _ddi_files]
}).to_csv("./DDI_predictions.csv", index=False)
# end change
