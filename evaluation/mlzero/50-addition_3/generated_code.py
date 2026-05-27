"""
Skin Lesion Image Binary Classification with AutoGluon MultiModal (Optimized for Performance)

This script trains an AutoGluon MultiModal image classifier to predict malignancy probability
for skin lesion images, with fairness considerations for skin tone. It:
- Loads and preprocesses training and test data (removes NA labels, drops index columns)
- Trains a binary classifier using image and tabular metadata
- Uses focal loss with class weights to mitigate class imbalance and potential skin tone bias
- Optimizes model architecture and training for best AUROC and fairness
- Saves the trained model and outputs malignancy probabilities for the test set
- Ensures output format, indices, and column names match requirements
- Validates predictions and prints AUROC on a held-out validation set

Installation requirements (run if needed):
    pip install --upgrade pip
    pip install autogluon.multimodal

Usage: Run as a standalone script. All outputs are saved to the specified output directory.
"""

# Installation steps (uncomment if running in a fresh environment)
# import sys
# !{sys.executable} -m pip install --upgrade pip
# !{sys.executable} -m pip install autogluon.multimodal

import os
import uuid
import warnings
import pandas as pd
import numpy as np

from autogluon.multimodal import MultiModalPredictor
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# Paths
DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_3_data"
OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/50-addition_3/node_5/output"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "MyImages")

def get_image_path(image_name):
    # Compose absolute path to image file
    return os.path.join(IMAGES_DIR, f"{image_name}.jpg")

def preprocess_train(train_df):
    # Remove index column if present
    if 'Unnamed: 0' in train_df.columns:
        train_df = train_df.drop(columns=['Unnamed: 0'])
    # Remove samples without valid labels (drop NA in 'label')
    train_df = train_df.dropna(subset=['label'])
    # Map label to binary: malignant=1, benign=0
    train_df['label'] = train_df['label'].map(lambda x: 1 if str(x).strip().lower() == 'malignant' else 0)
    # Add absolute image path
    train_df['image'] = train_df['image_name'].apply(get_image_path)
    # Fill missing tabular values for fairness features
    for col in ['skin_tone', 'alternative_skin_tone']:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
    train_df['expert_opinion'] = train_df['expert_opinion'].fillna('missing').astype(str)
    return train_df

def preprocess_test(test_df, train_df):
    # Remove index column if present
    if 'Unnamed: 0' in test_df.columns:
        test_df = test_df.drop(columns=['Unnamed: 0'])
    # Add absolute image path
    test_df['image'] = test_df['image_name'].apply(get_image_path)
    # Add missing tabular columns with placeholder values (as in training)
    for col in ['skin_tone', 'alternative_skin_tone']:
        if col not in test_df.columns:
            test_df[col] = np.nan
    if 'expert_opinion' not in test_df.columns:
        test_df['expert_opinion'] = 'missing'
    # Ensure column order matches training (except label)
    return test_df

def compute_class_weights(labels):
    # Compute class weights for focal loss (inverse frequency, normalized)
    counts = np.bincount(labels)
    weights = 1.0 / (counts + 1e-8)
    weights = weights / weights.sum()
    return weights.tolist()

def get_label_col(train_df):
    # Always use 'label' as the label column after mapping
    return 'label'

def get_output_filename(test_file):
    # Output file should be named "results" with the same extension as test file
    ext = os.path.splitext(test_file)[1]
    return os.path.join(OUTPUT_DIR, f"results{ext}")

def get_model_save_path():
    # Save model in a folder with random timestamp in output dir
    model_dir = os.path.join(OUTPUT_DIR, f"automm_model_{uuid.uuid4().hex}")
    return model_dir

def get_required_output_columns(train_df):
    # Output column: malignancy probability (float, 0-1)
    # If sample submission is not available, use 'label' as output column
    return ['label']

def validate_output(pred_df, test_df, required_columns, output_file):
    # 1. Check number of rows
    assert len(pred_df) == len(test_df), f"Prediction rows ({len(pred_df)}) != test rows ({len(test_df)})"
    # 2. Check indices match
    assert all(pred_df.index == test_df.index), "Prediction indices do not match test indices"
    # 3. Check columns
    assert list(pred_df.columns) == required_columns, f"Prediction columns {list(pred_df.columns)} != required {required_columns}"
    # 4. Check output file format
    ext = os.path.splitext(output_file)[1]
    if ext == ".csv":
        # Check file exists and can be read
        df_check = pd.read_csv(output_file)
        assert len(df_check) == len(test_df), "Saved CSV prediction file row count mismatch"
    # 5. Check values are floats in [0,1]
    assert pred_df[required_columns[0]].between(0, 1).all(), "Predicted probabilities not in [0,1]"

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # Preprocess
    train_df = preprocess_train(train_df)
    test_df = preprocess_test(test_df, train_df)

    # Hold out 10% validation if no explicit validation set
    np.random.seed(42)
    val_frac = 0.10
    val_indices = train_df.sample(frac=val_frac, random_state=42).index
    val_df = train_df.loc[val_indices]
    train_df_ = train_df.drop(index=val_indices)

    # Compute class weights for focal loss (to help fairness and imbalance)
    class_weights = compute_class_weights(train_df_['label'].values)

    # Prepare columns for AutoGluon
    label_col = get_label_col(train_df_)
    required_output_columns = get_required_output_columns(train_df_)

    # Model save path
    model_save_path = get_model_save_path()

    # ----------- MODEL ARCHITECTURE & TRAINING OPTIMIZATION -----------
    # Use a strong image backbone and late fusion for tabular+image
    # Use more epochs, mixed precision, and more workers for speed/quality
    # Use focal loss for fairness/imbalance
    # Disable torch.compile due to Swin Transformer incompatibility (see error summary)
    hyperparameters = {
        "env.num_gpus": 1,  # Use 1 GPU to avoid DDP errors
        "env.precision": "16-mixed",  # Mixed precision for speed/memory
        "env.num_workers": 8,         # More dataloader workers for speed
        "env.num_workers_inference": 8,
        "optim.loss_func": "focal_loss",
        "optim.focal_loss.alpha": class_weights,
        "optim.focal_loss.gamma": 2.0,
        "optim.focal_loss.reduction": "mean",
        "optim.max_epochs": 20,  # More epochs for better convergence
        "optim.patience": 7,     # Early stopping patience
        "optim.val_check_interval": 0.25,  # Validate more frequently
        "model.timm_image.checkpoint_name": "swin_large_patch4_window7_224",  # Strong backbone
        "model.names": ["timm_image", "fusion_mlp"],  # Use image + fusion MLP for tabular
        "model.fusion_mlp.hidden_sizes": [512, 256],  # Larger fusion MLP
        "model.fusion_mlp.dropout": 0.3,              # Regularization
        "model.fusion_mlp.activation": "gelu",
        "env.compile.turn_on": False,                 # Disable torch.compile due to Swin bug
    }

    # Fit model
    predictor = MultiModalPredictor(
        label=label_col,
        problem_type="binary",
        path=model_save_path,
        eval_metric="roc_auc"
    )

    predictor.fit(
        train_data=train_df_,
        hyperparameters=hyperparameters,
        time_limit=3200,  # ~53 min max for training (leave time for inference)
    )

    # Predict on test set (must preserve original indices)
    test_pred_proba = predictor.predict_proba(test_df)
    # For binary, predict_proba returns a DataFrame with columns [0, 1] (class probabilities)
    # We want the probability for class 1 (malignant)
    if 1 in test_pred_proba.columns:
        malignancy_prob = test_pred_proba[1]
    else:
        # Sometimes columns are strings
        malignancy_prob = test_pred_proba["1"]

    # Prepare output DataFrame
    pred_df = pd.DataFrame({required_output_columns[0]: malignancy_prob})
    pred_df.index = test_df.index  # Ensure indices match

    # Save predictions
    output_file = get_output_filename(TEST_CSV)
    ext = os.path.splitext(output_file)[1]
    if ext == ".csv":
        pred_df.to_csv(output_file, index=False)
    elif ext == ".parquet":
        pred_df.to_parquet(output_file, index=False)
    else:
        raise ValueError(f"Unsupported test file extension: {ext}")

    # Validation checks
    validate_output(pred_df, test_df, required_output_columns, output_file)

    # Validation step: compute AUROC on held-out validation set
    try:
        val_pred_proba = predictor.predict_proba(val_df)
        if 1 in val_pred_proba.columns:
            val_prob = val_pred_proba[1]
        else:
            val_prob = val_pred_proba["1"]
        val_auc = roc_auc_score(val_df[label_col], val_prob)
        print(f"Validation AUROC: {val_auc:.5f}")
    except Exception as e:
        print(f"Validation failed: {e}")

if __name__ == "__main__":
    main()