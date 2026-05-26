"""
Skin Lesion Malignancy Prediction Script using AutoGluon MultiModal

This script trains an image classification model to predict the probability of malignancy for skin lesion images.
It uses AutoGluon MultiModal for binary classification (malignant vs. benign), with fairness consideration towards skin tone.
The script:
- Loads and preprocesses the data (removes NA labels, drops index columns, constructs absolute image paths)
- Trains a model (with 10% validation split if no validation set is provided)
- Saves the trained model to a timestamped folder in the specified output directory
- Predicts malignancy probabilities for the test set, preserving original indices and output format
- Saves predictions in the same format as the test file, named "results"
- Provides a function to predict malignancy probabilities for any folder of new images
- Performs validation checks on the output
- Prints AUROC on the held-out validation set (if possible)

# Installation (run in bash script before running this Python script):
# pip install --upgrade pip
# pip install autogluon.multimodal

Author: AutoML Agent
"""

import os
import uuid
import warnings
import pandas as pd
import numpy as np
from datetime import datetime

warnings.filterwarnings('ignore')

# ========== CONFIGURATION ==========
# start change
# DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_1_data"  # original
DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/evaluation_data/"
# end change
IMG_DIR = os.path.join(DATA_DIR, "MyImages")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
# start change
# OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/38-addition_1/node_6/output"  # original
OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/mlzero/38-addition_1"
# end change
os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    # 1. Load Data
    train = pd.read_csv(TRAIN_CSV)
    # start change
    # test = pd.read_csv(TEST_CSV)
    test = pd.read_csv(TEST_CSV, dtype={"image_name": str})
    # end change

    # 2. Data Preprocessing

    # Remove unnecessary index column if present
    for df in [train, test]:
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)

    # Remove training samples without valid labels (drop NA in 'label')
    train = train.dropna(subset=['label'])

    # Map label to binary: malignant=1, benign=0
    train['label'] = train['label'].astype(str)
    train['label'] = train['label'].apply(lambda x: 1 if x.strip().lower() == 'malignant' else 0)

    # Construct absolute image paths for train and test
    def image_name_to_path(image_name):
        return os.path.abspath(os.path.join(IMG_DIR, f"{image_name}.jpg"))

    train['image'] = train['image_name'].apply(image_name_to_path)
    # start change
    # test['image'] = test['image_name'].apply(image_name_to_path)  # original (.jpg)
    test['image'] = test['image_name'].apply(
        lambda x: os.path.abspath(os.path.join(IMG_DIR, f"{x}.png"))
    )
    # end change

    # Ensure test set has all columns used in training except label
    # (AutoGluon MultiModal requires all columns except label at prediction time)
    train_cols = set(train.columns) - {'label'}
    for col in train_cols:
        if col not in test.columns:
            test[col] = np.nan
    # Ensure column order matches
    test = test[[col for col in train.columns if col != 'label']]

    # 3. Validation Split (10% holdout if no validation set provided)
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(
        train,
        test_size=0.1,
        random_state=42,
        stratify=train['label']
    )

    # 4. Model Training
    from autogluon.multimodal import MultiModalPredictor

    # Create a unique model directory with random timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    model_dir = os.path.join(OUTPUT_DIR, f"autogluon_model_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)

    # Use focal loss for fairness (helps with class imbalance)
    # Compute class weights for focal loss
    class_counts = train_data['label'].value_counts().sort_index()
    if 0 not in class_counts:
        class_counts[0] = 0
    if 1 not in class_counts:
        class_counts[1] = 0
    total = class_counts.sum()
    weights = [float(1.0 / (class_counts[0] / total + 1e-8)), float(1.0 / (class_counts[1] / total + 1e-8))]
    weights = [float(w) for w in (np.array(weights) / np.sum(weights))]  # Normalize as Python floats

    # Model architecture improvements: use a larger backbone and more epochs for better performance
    predictor = MultiModalPredictor(
        label="label",
        problem_type="binary",
        path=model_dir
    )

    predictor.fit(
        train_data=train_data,
        tuning_data=val_data,
        hyperparameters={
            "env.num_gpus": 1,
            "optim.loss_func": "focal_loss",
            "optim.focal_loss.alpha": weights,
            "optim.focal_loss.gamma": 2.0,
            "optim.focal_loss.reduction": "mean",
            # Use a larger backbone for better performance (e.g., Swin Transformer base)
            "model.timm_image.checkpoint_name": "swin_base_patch4_window7_224",
            "optim.max_epochs": 40,
            "env.per_gpu_batch_size": 16,  # Reasonable batch size for 1 GPU, adjust if OOM
            "env.num_workers": 8,          # Use more workers for faster data loading
        },
        time_limit=3200,  # ~53 minutes for training
        presets="best_quality"
    )

    # 5. Prediction on Test Set

    test_pred_df = test.copy()
    proba = predictor.predict_proba(test_pred_df)
    if 1 in proba.columns:
        malignancy_prob = proba[1].values
    elif '1' in proba.columns:
        malignancy_prob = proba['1'].values
    else:
        malignancy_prob = proba.iloc[:, -1].values

    # start change
    _ddi_df = pd.DataFrame({
        "DDI_file": test["image_name"].astype(str) + ".png",
        "predicted_probability": malignancy_prob.astype(float),
    }).reset_index(drop=True)
    _ddi_df.to_csv(os.path.join(OUTPUT_DIR, "DDI_predictions.csv"), index=False)
    # end change

    results_df = test[['image_name']].copy()
    results_df['label'] = malignancy_prob
    results_df.index = test.index  # Preserve original indices

    # Save results in the same format/extension as test.csv
    test_ext = os.path.splitext(TEST_CSV)[1].lower()
    results_path = os.path.join(OUTPUT_DIR, f"results{test_ext}")
    if test_ext == ".csv":
        results_df.to_csv(results_path, index=True)
    elif test_ext in [".tsv", ".txt"]:
        results_df.to_csv(results_path, sep='\t', index=True)
    else:
        results_df.to_csv(results_path, index=True)

    # 6. Validation Checks

    loaded_results = pd.read_csv(results_path, index_col=0 if 'Unnamed: 0' in test.columns else None)
    assert len(loaded_results) == len(test), f"Prediction file row count {len(loaded_results)} != test set {len(test)}"
    assert all(loaded_results.index == test.index), "Prediction file indices do not match test set indices"
    required_cols = ['image_name', 'label']
    assert all([col in loaded_results.columns for col in required_cols]), f"Prediction file missing required columns: {required_cols}"
    assert os.path.splitext(results_path)[1].lower() == test_ext, "Output file extension does not match test file"
    assert np.all((loaded_results['label'] >= 0) & (loaded_results['label'] <= 1)), "Predicted probabilities not in [0,1]"

    print(f"Predictions saved to: {results_path}")

    # 7. Validation Metric (AUROC on held-out validation set)
    try:
        from sklearn.metrics import roc_auc_score
        val_pred_proba = predictor.predict_proba(val_data)
        if 1 in val_pred_proba.columns:
            val_malignancy_prob = val_pred_proba[1].values
        elif '1' in val_pred_proba.columns:
            val_malignancy_prob = val_pred_proba['1'].values
        else:
            val_malignancy_prob = val_pred_proba.iloc[:, -1].values
        val_auc = roc_auc_score(val_data['label'], val_malignancy_prob)
        print(f"Validation AUROC: {val_auc:.4f}")
    except Exception as e:
        print(f"Validation AUROC computation failed: {e}")

    # 8. Provide function for new image folder prediction
    def predict_folder(folder_path, model_path=model_dir):
        """
        Predict malignancy probability for all .jpg images in a folder.

        Args:
            folder_path (str): Path to folder containing images.
            model_path (str): Path to saved AutoGluon model.

        Returns:
            pd.DataFrame: DataFrame with columns ['image_name', 'label'] (malignancy probability)
        """
        from autogluon.multimodal import MultiModalPredictor
        import glob

        image_files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
        if not image_files:
            raise ValueError(f"No .jpg images found in {folder_path}")

        df = pd.DataFrame({
            'image_name': [os.path.splitext(os.path.basename(f))[0] for f in image_files],
            'image': [os.path.abspath(f) for f in image_files]
        })
        # Add missing columns with NaN to match training columns (except label)
        for col in train_cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df[[col for col in train.columns if col != 'label']]

        predictor = MultiModalPredictor.load(model_path)
        proba = predictor.predict_proba(df)
        if 1 in proba.columns:
            malignancy_prob = proba[1].values
        elif '1' in proba.columns:
            malignancy_prob = proba['1'].values
        else:
            malignancy_prob = proba.iloc[:, -1].values

        result = pd.DataFrame({
            'image_name': df['image_name'],
            'label': malignancy_prob
        })
        return result

    # Example usage (commented out):
    # folder_results = predict_folder("/path/to/new/images")
    # print(folder_results.head())

    print("Script completed successfully.")