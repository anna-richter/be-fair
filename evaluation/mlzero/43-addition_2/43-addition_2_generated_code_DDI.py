"""
Skin Lesion Malignancy Prediction Script using AutoGluon MultiModal

This script performs binary image classification (malignant vs. benign) on skin lesion images.
It:
- Loads and preprocesses the data (removes NA labels, drops index columns)
- Trains an AutoGluon MultiModalPredictor on the training set (with 10% held out for validation)
- Predicts malignancy probability (float in [0,1]) for each test image, preserving test indices
- Saves predictions in the same format as test.csv, with correct column names and order
- Saves the trained model to a timestamped folder in the output directory
- Performs validation checks on the output file
- Prints AUROC on the held-out validation set (if possible)

Installation requirements (run if needed):
    # pip install --upgrade pip
    # pip install autogluon.multimodal

Data paths and output directory are hardcoded for this challenge.
"""

import os
import uuid
import warnings
import pandas as pd
import numpy as np
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # ========== CONFIGURATION ==========
    # start change
    # DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_2_data"  # original
    DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/evaluation_data/"
    # end change
    IMG_DIR = os.path.join(DATA_DIR, "MyImages")
    TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
    TEST_CSV = os.path.join(DATA_DIR, "test.csv")
    # start change
    # OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/43-addition_2/node_11/output"  # original
    OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/mlzero/43-addition_2"
    # end change
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    MODEL_SAVE_DIR = os.path.join(
        OUTPUT_DIR, f"autogluon_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    )
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # 1. Load data
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    # 2. Data preprocessing
    # Remove unnecessary index column if present
    for df in [train, test]:
        for idx_col in ["Unnamed: 0", "index"]:
            if idx_col in df.columns:
                df.drop(columns=[idx_col], inplace=True)

    # Remove training samples without valid labels (drop NA in label)
    train = train.dropna(subset=["label"])

    # Only keep relevant columns for training
    def image_name_to_path(name):
        return os.path.abspath(os.path.join(IMG_DIR, f"{name}.jpg"))

    train["image"] = train["image_name"].apply(image_name_to_path)
    # start change
    # test["image"] = test["image_name"].apply(image_name_to_path)  # original (.jpg)
    test["image"] = test["image_name"].apply(
        lambda x: os.path.abspath(os.path.join(IMG_DIR, f"{x}.png"))
    )
    # end change

    # Only keep columns needed for training
    train_for_fit = train[["image", "label"]].copy()

    # 3. Label normalization: Map all non-benign labels to 'malignant' or 'benign'
    label_map = {
        "malignant": "malignant",
        "benign": "benign",
        "non-neoplastic": "benign",  # treat as benign
        "neoplastic": "malignant",   # treat as malignant if present
        "cancer": "malignant",       # treat as malignant if present
        "non-cancer": "benign",      # treat as benign if present
    }
    train_for_fit["label"] = train_for_fit["label"].map(lambda x: label_map.get(str(x).strip().lower(), x))

    # Sanity check: Only two classes
    unique_labels = sorted(train_for_fit["label"].unique())
    assert set(unique_labels) <= {"malignant", "benign"}, f"Unexpected labels: {unique_labels}"

    # 4. Class imbalance handling (optional): Use focal loss with class weights if imbalance is severe
    class_counts = train_for_fit["label"].value_counts()
    classes = ["benign", "malignant"]
    class_weights = []
    for c in classes:
        count = class_counts.get(c, 0)
        if count == 0:
            class_weights.append(0.0)
        else:
            class_weights.append(1.0 / count)
    class_weights = np.array(class_weights)
    class_weights = class_weights / class_weights.sum()  # Normalize to sum to 1

    # 5. Split train/validation (10% holdout)
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(splitter.split(train_for_fit, train_for_fit["label"]))
    train_data = train_for_fit.iloc[train_idx].reset_index(drop=True)
    val_data = train_for_fit.iloc[val_idx].reset_index(drop=True)

    # 6. Model training
    from autogluon.multimodal import MultiModalPredictor

    predictor = MultiModalPredictor(
        label="label",
        problem_type="classification",
        path=MODEL_SAVE_DIR,
    )

    # Set up hyperparameters (prioritize model architecture and training optimization)
    hyperparameters = {
        "model.timm_image.checkpoint_name": "convnext_base",  # Stronger backbone than Swin-Tiny
        "env.num_gpus": 1,  # Use single GPU
        "optim.loss_func": "focal_loss",
        "optim.focal_loss.alpha": class_weights.tolist(),
        "optim.focal_loss.gamma": 2.0,
        "optim.focal_loss.reduction": "mean",
        "optim.max_epochs": 20,  # Slightly longer training for better convergence
        "env.precision": "16-mixed",  # Mixed precision for speed/memory
        "env.num_workers": 8,
        "env.num_workers_inference": 4,
        "optim.patience": 5,
        "optim.val_check_interval": 0.25,
        "optim.gradient_clip_val": 2.0,
        "optim.gradient_clip_algorithm": "norm",
        "optim.top_k": 3,
        "optim.top_k_average_method": "greedy_soup",
    }

    # Fit model (use a valid preset string)
    predictor.fit(
        train_data=train_data,
        tuning_data=val_data,
        hyperparameters=hyperparameters,
        seed=42,
        time_limit=3600 - 600,  # Leave 10min for inference/validation
        presets="high_quality",  # Use a strong preset for best performance
    )

    # 7. Prediction on test set
    test_pred_input = test[["image"]].copy()
    proba = predictor.predict_proba(test_pred_input)
    # proba is a DataFrame with columns ['benign', 'malignant']

    # Output: ['image_name', 'label'] where 'label' is the malignancy probability (float)
    output_df = test[["image_name"]].copy()
    output_df["label"] = proba["malignant"].astype(float)

    # start change
    _ddi_df = pd.DataFrame({
        "DDI_file": test["image_name"].astype(str) + ".png",
        "predicted_probability": proba["malignant"].astype(float).values,
    }).reset_index(drop=True)
    _ddi_df.to_csv(os.path.join(OUTPUT_DIR, "DDI_predictions.csv"), index=False)
    # end change

    # Ensure output has the same number of rows and indices as test set
    assert len(output_df) == len(test), "Prediction output row count does not match test set"

    # Save results in the same format/extension as test.csv
    output_ext = os.path.splitext(TEST_CSV)[1].lower()
    results_path = os.path.join(OUTPUT_DIR, f"results{output_ext}")
    output_df.to_csv(results_path, index=False)

    # 8. Save model (already saved by predictor.fit, but ensure it's in the output dir)
    # (AutoGluon saves to MODEL_SAVE_DIR automatically)

    # 9. Validation: Compute AUROC on held-out validation set
    try:
        from sklearn.metrics import roc_auc_score

        val_pred_input = val_data[["image"]].copy()
        val_proba = predictor.predict_proba(val_pred_input)
        y_true = val_data["label"].map({"benign": 0, "malignant": 1}).values
        y_score = val_proba["malignant"].values
        val_auroc = roc_auc_score(y_true, y_score)
        print(f"Validation AUROC: {val_auroc:.5f}")
    except Exception as e:
        print(f"Validation failed: {e}")

    # 10. Output validation checks
    assert os.path.exists(results_path), f"Results file not found at {results_path}"
    pred_df = pd.read_csv(results_path)
    expected_cols = ["image_name", "label"]
    assert list(pred_df.columns) == expected_cols, f"Output columns {list(pred_df.columns)} do not match {expected_cols}"
    assert len(pred_df) == len(test), f"Output row count {len(pred_df)} does not match test set {len(test)}"
    assert all(pred_df["image_name"] == test["image_name"]), "Test image_name order mismatch"
    assert np.all((pred_df["label"] >= 0) & (pred_df["label"] <= 1)), "Predicted probabilities not in [0,1]"

    print(f"Prediction file saved to: {results_path}")
    print(f"Model saved to: {MODEL_SAVE_DIR}")