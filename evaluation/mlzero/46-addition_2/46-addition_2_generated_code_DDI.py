"""
Skin Lesion Malignancy Probability Prediction (Image Classification, CPU-only, Improved)

This script trains a machine learning model to predict the probability of malignancy for skin lesion images.
It uses a hybrid approach:
- Extracts features from images using an ensemble of pretrained CNNs (ResNet50 and EfficientNet-B0, both from torchvision, avgpool outputs).
- Concatenates these image features with available tabular features (skin_tone, alternative_skin_tone), ensuring feature consistency between train and test.
- Trains a LightGBM classifier (with class weights for fairness, optimized for AUROC).
- Saves the trained model to a timestamped folder in the specified output directory.
- Predicts malignancy probabilities for the test set, preserving original indices and format.
- Saves predictions in the same format and extension as the test file, with correct column names.
- Performs validation checks to ensure output integrity.
- If training labels are available, holds out 10% of training data for validation and prints AUROC.

# Installation (run in bash before this script):
# pip install pandas numpy scikit-learn lightgbm pillow tqdm torch torchvision

Author: AutoML Agent
"""

import os
import uuid
import pandas as pd
import numpy as np
from datetime import datetime

def get_image_path(image_name, image_dir):
    """Return absolute path to image file given image_name (without extension)."""
    return os.path.abspath(os.path.join(image_dir, f"{image_name}.jpg"))

def map_label_to_binary(label):
    """Map string label to binary: malignant=1, non-neoplastic=0."""
    if pd.isna(label):
        return np.nan
    label = str(label).strip().lower()
    if label == "malignant":
        return 1
    elif label == "non-neoplastic":
        return 0
    else:
        return np.nan

def infer_test_file_format(test_csv_path):
    """Infer file extension and separator from test file."""
    _, ext = os.path.splitext(test_csv_path)
    ext = ext.lower()
    if ext == ".csv":
        sep = ","
    elif ext == ".tsv":
        sep = "\t"
    else:
        raise ValueError(f"Unknown test file extension: {ext}")
    return ext, sep

def get_result_file_path(test_csv_path, output_dir):
    """Return output file path for results, matching test file extension."""
    ext, _ = infer_test_file_format(test_csv_path)
    return os.path.join(output_dir, f"results{ext}")

def get_sample_submission_columns():
    """Return the correct output column name(s) for predictions."""
    return ["malignancy"]

def prepare_train_data(train_df, image_dir):
    """Preprocess train data: drop NA labels, drop index, map labels, add image path."""
    train_df = train_df.dropna(subset=["label"]).copy()
    if "Unnamed: 0" in train_df.columns:
        train_df = train_df.drop(columns=["Unnamed: 0"])
    train_df["malignancy"] = train_df["label"].map(map_label_to_binary)
    train_df = train_df.dropna(subset=["malignancy"])
    train_df["malignancy"] = train_df["malignancy"].astype(int)
    train_df["image"] = train_df["image_name"].apply(lambda x: get_image_path(x, image_dir))
    return train_df

def prepare_test_data(test_df, image_dir):
    """Preprocess test data: drop index, add image path. DO NOT drop any rows."""
    if "Unnamed: 0" in test_df.columns:
        test_df = test_df.drop(columns=["Unnamed: 0"])
    # start change
    # test_df["image"] = test_df["image_name"].apply(lambda x: get_image_path(x, image_dir))  # original (.jpg)
    test_df["image"] = test_df["image_name"].apply(
        lambda x: os.path.abspath(os.path.join(image_dir, f"{x}.png"))
    )
    # end change
    return test_df

def save_predictions_with_indices(test_df, preds, output_path, output_col, sep):
    """Save predictions to output_path, preserving original indices and format."""
    out_df = test_df.copy()
    out_df[output_col[0]] = preds
    columns_to_save = []
    if "Unnamed: 0" in test_df.columns:
        columns_to_save.append("Unnamed: 0")
    columns_to_save.append("image_name")
    columns_to_save += output_col
    columns_to_save = [col for col in columns_to_save if col in out_df.columns]
    out_df = out_df[columns_to_save]
    out_df.to_csv(output_path, index=False, sep=sep)

def compute_class_weights(y):
    """Compute class weights for binary classification (malignant/benign)."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    return dict(zip(classes, weights))

def compute_auroc(y_true, y_pred):
    """Compute AUROC score."""
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_pred)

def extract_features_ensemble(image_paths, batch_size=64):
    """
    Extract features from images using an ensemble of pretrained CNNs (ResNet50 and EfficientNet-B0).
    Returns a numpy array of shape (n_images, 2048+1280).
    """
    from PIL import Image
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    from tqdm import tqdm

    device = torch.device("cpu")
    # ResNet50
    resnet = models.resnet50(pretrained=True)
    resnet.eval()
    resnet_feature = torch.nn.Sequential(*(list(resnet.children())[:-1]))
    resnet_feature.to(device)
    # EfficientNet-B0
    effnet = models.efficientnet_b0(pretrained=True)
    effnet.eval()
    effnet_feature = torch.nn.Sequential(*(list(effnet.children())[:-1]))
    effnet_feature.to(device)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    features = []
    n = len(image_paths)
    for i in tqdm(range(0, n, batch_size), desc="Extracting image features (ResNet50+EffNetB0)"):
        batch_paths = image_paths[i:i+batch_size]
        batch_imgs = []
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                img = preprocess(img)
            except Exception:
                img = torch.zeros(3, 224, 224)
            batch_imgs.append(img)
        batch_tensor = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            res_feats = resnet_feature(batch_tensor)
            res_feats = res_feats.view(res_feats.size(0), -1).cpu().numpy()
            eff_feats = effnet_feature(batch_tensor)
            eff_feats = eff_feats.view(eff_feats.size(0), -1).cpu().numpy()
            feats = np.concatenate([res_feats, eff_feats], axis=1)
        features.append(feats)
    features = np.concatenate(features, axis=0)
    return features

def get_tabular_features(df, tabular_cols, fill_values=None):
    """
    Extract tabular features (skin_tone, alternative_skin_tone) as float32.
    If fill_values is provided, fill missing columns with those values.
    """
    feats = []
    for col in tabular_cols:
        if col in df.columns:
            feats.append(df[col].astype(np.float32).values.reshape(-1, 1))
        else:
            # Fill with provided value or zero if not specified
            fill_val = 0.0 if fill_values is None else fill_values.get(col, 0.0)
            feats.append(np.full((len(df), 1), fill_val, dtype=np.float32))
    if feats:
        return np.concatenate(feats, axis=1)
    else:
        return None

if __name__ == "__main__":
    # Constants
    # start change
    # DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_2_data"  # original
    DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/evaluation_data/"
    # end change
    # start change
    # OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/46-addition_2/node_4/output"  # original
    OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/mlzero/46-addition_2"
    # end change
    IMAGE_DIR = os.path.join(DATA_DIR, "MyImages")
    TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
    TEST_CSV = os.path.join(DATA_DIR, "test.csv")

    # 1. Load data
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # 2. Preprocess data
    train_df = prepare_train_data(train_df, IMAGE_DIR)
    test_df = prepare_test_data(test_df, IMAGE_DIR)

    # 3. Validation split (if labeled training data is available)
    do_validation = False
    if len(train_df) > 0:
        do_validation = True
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
        val_frac = 0.10
        val_size = int(np.ceil(len(train_df) * val_frac))
        val_df = train_df.iloc[:val_size].reset_index(drop=True)
        train_df_fit = train_df.iloc[val_size:].reset_index(drop=True)
    else:
        train_df_fit = train_df
        val_df = None

    # 4. Extract image features (ResNet50 + EfficientNetB0, CPU)
    print("Extracting features for training images...")
    X_img_train = extract_features_ensemble(train_df_fit["image"].tolist())
    y_train = train_df_fit["malignancy"].values

    # Tabular columns to use (ensure consistent order and presence)
    tabular_cols = ["skin_tone", "alternative_skin_tone"]
    # Compute fill values (mean from training set) for missing columns in test
    fill_values = {}
    for col in tabular_cols:
        if col in train_df_fit.columns:
            fill_values[col] = train_df_fit[col].astype(np.float32).mean()
        else:
            fill_values[col] = 0.0

    X_tab_train = get_tabular_features(train_df_fit, tabular_cols)
    if X_tab_train is not None:
        X_train = np.concatenate([X_img_train, X_tab_train], axis=1)
    else:
        X_train = X_img_train

    if do_validation and val_df is not None and len(val_df) > 0:
        print("Extracting features for validation images...")
        X_img_val = extract_features_ensemble(val_df["image"].tolist())
        y_val = val_df["malignancy"].values
        X_tab_val = get_tabular_features(val_df, tabular_cols, fill_values=fill_values)
        if X_tab_val is not None:
            X_val = np.concatenate([X_img_val, X_tab_val], axis=1)
        else:
            X_val = X_img_val

    print("Extracting features for test images...")
    X_img_test = extract_features_ensemble(test_df["image"].tolist())
    X_tab_test = get_tabular_features(test_df, tabular_cols, fill_values=fill_values)
    if X_tab_test is not None:
        X_test = np.concatenate([X_img_test, X_tab_test], axis=1)
    else:
        X_test = X_img_test

    # 5. Train classifier (LightGBM, with class weights, optimized for AUROC)
    from lightgbm import LGBMClassifier

    class_weights = compute_class_weights(y_train)
    clf = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        class_weight=class_weights,
        n_jobs=32,  # Use many CPUs for speed
        random_state=42,
        verbose=-1,
        boosting_type='gbdt',
        objective='binary',
        metric='auc'
    )
    clf.fit(X_train, y_train)

    # 6. Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    model_dir = os.path.join(OUTPUT_DIR, f"ml_model_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    import joblib
    joblib.dump(clf, os.path.join(model_dir, "model.lgbm"))
    with open(os.path.join(model_dir, "feature_info.txt"), "w") as f:
        f.write("ResNet50+EfficientNetB0 avgpool features, tabular features (skin_tone, alternative_skin_tone)\n")

    # 7. Predict on test set
    proba = clf.predict_proba(X_test)[:, 1]  # Probability of class 1 (malignant)

    # start change
    _ddi_df = pd.DataFrame({
        "DDI_file": test_df["image_name"].astype(str) + ".png",
        "predicted_probability": proba.astype(float),
    }).reset_index(drop=True)
    _ddi_df.to_csv(os.path.join(OUTPUT_DIR, "DDI_predictions.csv"), index=False)
    # end change

    # 8. Save predictions
    ext, sep = infer_test_file_format(TEST_CSV)
    output_col = get_sample_submission_columns()
    result_file = get_result_file_path(TEST_CSV, OUTPUT_DIR)
    save_predictions_with_indices(test_df, proba, result_file, output_col, sep)

    # 9. Validation step (if applicable)
    if do_validation and val_df is not None and len(val_df) > 0:
        try:
            val_proba = clf.predict_proba(X_val)[:, 1]
            val_score = compute_auroc(y_val, val_proba)
            print(f"Validation AUROC: {val_score:.5f}")
        except Exception as e:
            print(f"Validation failed: {e}")

    # 10. Validation checks
    assert os.path.exists(result_file), f"Result file not found: {result_file}"
    pred_df = pd.read_csv(result_file, sep=sep)
    assert len(pred_df) == len(test_df), "Number of predictions does not match number of test samples."
    if "Unnamed: 0" in test_df.columns:
        assert all(pred_df["Unnamed: 0"].values == test_df["Unnamed: 0"].values), "Test indices not preserved."
    for col in output_col:
        assert col in pred_df.columns, f"Missing output column: {col}"
    preds = pred_df[output_col[0]].values
    assert np.all((preds >= 0) & (preds <= 1)), "Predictions are not in [0, 1]."
    _, test_ext = os.path.splitext(TEST_CSV)
    _, result_ext = os.path.splitext(result_file)
    assert test_ext == result_ext, "Output file extension does not match test file."

    print(f"Predictions saved to: {result_file}")
    print("All validation checks passed.")