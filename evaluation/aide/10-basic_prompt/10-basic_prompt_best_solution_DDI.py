import os
import pandas as pd
import numpy as np
from PIL import Image
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score


class ImageDataset(Dataset):
    def __init__(self, image_names, image_dir, transform):
        self.names = image_names
        self.dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
# --- BEGIN CHANGE ---
#        path = os.path.join(self.dir, name + ".jpg")
        path = os.path.join(self.dir, name + ".png")
# --- END CHANGE ---
        img = Image.open(path).convert("RGB")
        return self.transform(img), name


def extract_features(model, loader, device):
    model.eval()
    feats, names = [], []
    with torch.no_grad():
        for imgs, batch_names in loader:
            imgs = imgs.to(device)
            f = model(imgs).cpu().numpy()
            feats.append(f)
            names.extend(batch_names)
    return np.vstack(feats), names


def predict(image_folder: str, model_path: str = "working/pipeline.pkl"):
    """
    Predict malignancy probabilities for .jpg images in a folder.
    Args:
        image_folder (str): Directory containing .jpg images
        model_path (str): Path to saved pipeline (scaler+logistic)
    Returns:
        pandas.DataFrame with ['image_name','malignant_prob']
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # EfficientNet-B0 backbone
    backbone = models.efficientnet_b0(pretrained=True)
    backbone.classifier = torch.nn.Identity()
    backbone.to(device).eval()
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
# --- BEGIN CHANGE ---
    """
    imgs = [
        os.path.splitext(f)[0]
        for f in os.listdir(image_folder)
        if f.lower().endswith(".jpg")
    ]
    """
    imgs = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(image_folder)
        if f.lower().endswith(".png")
    )
# --- END CHANGE ---
    ds = ImageDataset(imgs, image_folder, tfm)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    feats, names = extract_features(backbone, loader, device)
    pipeline = joblib.load(model_path)
    probs = pipeline.predict_proba(feats)[:, 1]
# --- BEGIN CHANGE ---
#    return pd.DataFrame({"image_name": names, "malignant_prob": probs})
    return pd.DataFrame({"DDI_file": [n + ".png" for n in names], "predicted_probability": probs})
# --- END CHANGE ---


if __name__ == "__main__":
    """
    # Paths
    csv_path = "./input/mydataset.csv"
    image_dir = "./input/MyImages"
    working_dir = "./working"
    os.makedirs(working_dir, exist_ok=True)

    # Load labels
    df = pd.read_csv(csv_path)
    df["target"] = (df["label"] == "malignant").astype(int)
    image_names = df["image_name"].tolist()
    y_full = df["target"].values

    # Transforms & device
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # EfficientNet-B0 backbone for feature extraction
    backbone = models.efficientnet_b0(pretrained=True)
    backbone.classifier = torch.nn.Identity()
    backbone.to(device)

    # Extract features for all images
    ds_all = ImageDataset(image_names, image_dir, tfm)
    loader_all = DataLoader(ds_all, batch_size=64, shuffle=False, num_workers=4)
    X_full, names_order = extract_features(backbone, loader_all, device)
    # Align targets
    name_to_target = dict(zip(df["image_name"], df["target"]))
    y = np.array([name_to_target[n] for n in names_order])

    # Build pipeline and grid search
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs", max_iter=1000, class_weight="balanced"
                ),
            ),
        ]
    )
    param_grid = {"clf__C": [0.001, 0.01, 0.1, 1, 10]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=1)
    gs.fit(X_full, y)
    print(f"Best CV AUROC: {gs.best_score_:.4f}, Best C: {gs.best_params_['clf__C']}")

    # Save final pipeline
    joblib.dump(gs.best_estimator_, os.path.join(working_dir, "pipeline.pkl"))

    # If test folder exists, generate submission.csv
    test_folder = "./input/test"
    if os.path.isdir(test_folder):
        sub = predict(test_folder, os.path.join(working_dir, "pipeline.pkl"))
        sub.to_csv(os.path.join(working_dir, "submission.csv"), index=False)
    """

# --- BEGIN CHANGE ---
if __name__ == "__main__":
    ddi_image_dir = os.path.join("..", "..", "DDI", "images")
    predictions = predict(ddi_image_dir, model_path=os.path.join("working", "pipeline.pkl"))
    predictions.to_csv("10-basic_prompt_DDI_predictions.csv", index=False)
    print(f"Saved {len(predictions)} predictions to 10-basic_prompt_DDI_predictions.csv")
# --- END CHANGE ---
