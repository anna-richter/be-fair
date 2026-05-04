"""
Skin Lesion Image Binary Classification with AutoGluon MultiModal (Enhanced)
============================================================================
This script trains an image classification model using AutoGluon MultiModal on a dermatology dataset
to predict malignancy of skin lesions. It prioritizes model architecture and training optimization
by using strong image backbones, leveraging all available tabular features, and using advanced presets.

It:
- Loads and preprocesses the data (removes NA labels, drops index column)
- Trains a binary classifier (malignant vs. non-neoplastic) using both image and tabular features
- Uses a strong backbone (e.g., swin_transformer, vit, or efficientnet) and advanced presets for best performance
- Saves the trained model to a timestamped folder in the output directory
- Provides a function to predict malignancy probability (0–1) for new images in a folder
- Makes predictions on the test set, preserving original indices and output format
- Validates output file for correct format, column names, and row count
- Computes AUROC on a held-out validation set if possible

Installation requirements (run if needed):
# pip install autogluon.multimodal scikit-learn pandas

Author: AutoML Agent
"""

import os
import time
import random
import pandas as pd
import numpy as np

from autogluon.multimodal import MultiModalPredictor
from sklearn.metrics import roc_auc_score

# Set paths
DATA_DIR = "/home/anri21/be-fair/mlzero/addition_1_data"
OUTPUT_DIR = "/home/anri21/be-fair/mlzero/35-addition_1/node_3/output"
TRAIN_CSV = os.path.join(DATA_DIR, "mydataset.csv")
IMAGE_ROOT = DATA_DIR  # images are referenced by 'image_name' column, and are in DATA_DIR

RESULTS_FILENAME = "results"

def get_timestamp():
    """Generate a random timestamp for model folder naming."""
    return str(int(time.time())) + str(random.randint(1000, 9999))

def preprocess_train(df):
    """Preprocess training data: drop NA labels, drop index column if present."""
    df = df.dropna(subset=['label'])
    for idx_col in ['Unnamed: 0', 'index']:
        if idx_col in df.columns:
            df = df.drop(columns=[idx_col])
    return df

def preprocess_test(df):
    """Preprocess test data: drop index column if present, but DO NOT drop any rows."""
    for idx_col in ['Unnamed: 0', 'index']:
        if idx_col in df.columns:
            df = df.drop(columns=[idx_col])
    return df

def map_labels(df):
    """
    Map string labels to binary: 'malignant'->1, 'non-neoplastic'->0.
    If other labels exist, treat them as 0 (benign).
    """
    df = df.copy()
    df['label'] = df['label'].map(lambda x: 1 if str(x).strip().lower() == 'malignant' else 0)
    return df

def get_image_path_from_name(image_name):
    """Get absolute image path from image_name."""
    return os.path.join(IMAGE_ROOT, image_name)

def save_results(df, test_file, output_dir, filename="results"):
    """Save results in the same format and extension as test_file."""
    ext = os.path.splitext(test_file)[-1]
    out_path = os.path.join(output_dir, filename + ext)
    if ext == ".csv":
        df.to_csv(out_path, index=False)
    elif ext in [".parquet", ".pq"]:
        df.to_parquet(out_path, index=False)
    elif ext in [".tsv"]:
        df.to_csv(out_path, sep="\t", index=False)
    else:
        raise ValueError(f"Unsupported test file extension: {ext}")
    return out_path

def validate_output(pred_path, test_df, label_col):
    """Validation checks for output predictions."""
    pred_df = pd.read_csv(pred_path) if pred_path.endswith('.csv') else pd.read_parquet(pred_path)
    assert len(pred_df) == len(test_df), f"Prediction rows ({len(pred_df)}) != test rows ({len(test_df)})"
    assert label_col in pred_df.columns, f"Missing required column: {label_col}"
    assert np.all((pred_df[label_col] >= 0) & (pred_df[label_col] <= 1)), "Probabilities not in [0,1]"
    assert not pred_df[label_col].isnull().any(), "NaN values in predictions"
    print("Validation checks passed.")

def train_and_predict(train_df, test_df, image_col, label_col, output_dir, model_dir=None):
    """
    Train AutoGluon MultiModalPredictor and predict on test set.
    Returns: predictions DataFrame (with original test indices), model path, validation score (if available)
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    # Add absolute image paths
    train_df[image_col] = train_df[image_col].apply(get_image_path_from_name)
    test_df[image_col] = test_df[image_col].apply(get_image_path_from_name)

    # Use all available tabular features except label and image_name
    tabular_features = [col for col in train_df.columns if col not in [label_col, image_col]]

    # Model directory
    if model_dir is None:
        model_dir = os.path.join(output_dir, "model_" + get_timestamp())
    os.makedirs(model_dir, exist_ok=True)

    # Use strong image backbone and advanced presets for best performance
    predictor = MultiModalPredictor(
        label=label_col,
        problem_type="binary",
        path=model_dir,
        eval_metric="roc_auc",
    )
    # Use all features (image + tabular)
    fit_args = {
        "presets": "best_quality",  # best_quality uses strong backbones and heavy augmentation
        "hyperparameters": {
            "model.image": {"checkpoint_name": "swin_base_patch4_window7_224"},  # strong backbone
            # Optionally, you can add more advanced settings here
        },
        "time_limit": None,
    }
    # Fit model
    predictor.fit(
        train_df,
        **fit_args
    )

    # Validation: use predictor's internal validation if available
    val_score = None
    try:
        leaderboard = predictor.leaderboard(silent=True)
        if 'score_val' in leaderboard.columns:
            val_score = leaderboard['score_val'].iloc[0]
            print(f"AutoGluon-reported validation AUROC: {val_score:.4f}")
    except Exception as e:
        print(f"Validation failed: {e}")

    # Predict on test set (probabilities for class 1)
    test_probs = predictor.predict_proba(test_df, as_pandas=True)[1]  # Probability for class 1 (malignant)
    out_df = test_df.copy()
    out_df[label_col] = test_probs
    out_df = out_df[[image_col, label_col]]
    # Convert image path back to image_name for output
    out_df[image_col] = out_df[image_col].apply(lambda x: os.path.basename(x))
    return out_df, model_dir, val_score

def predict_on_folder(model_path, folder_path, image_col='image_name', label_col='label'):
    """
    Predict malignancy probability for all images in a folder.
    Returns a DataFrame with image_name and malignancy probability.
    """
    predictor = MultiModalPredictor.load(model_path)
    # List all image files in folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
    df = pd.DataFrame({image_col: [os.path.join(folder_path, f) for f in image_files]})
    # Predict probabilities
    probs = predictor.predict_proba(df, as_pandas=True)[1]
    result_df = pd.DataFrame({
        image_col: [os.path.basename(f) for f in df[image_col]],
        label_col: probs
    })
    return result_df

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load training data
    train_df = pd.read_csv(TRAIN_CSV)
    train_df = preprocess_train(train_df)
    train_df = map_labels(train_df)

    # For demonstration, let's use the same CSV as test (in real use, replace with actual test CSV)
    TEST_CSV = TRAIN_CSV  # Replace with actual test CSV path if available
    test_df = pd.read_csv(TEST_CSV)
    test_df = preprocess_test(test_df)
    # Remove label column if present (since test set should not have labels)
    if 'label' in test_df.columns:
        test_df = test_df.drop(columns=['label'])

    image_col = 'image_name'
    label_col = 'label'
    predictions_df, model_dir, val_score = train_and_predict(
        train_df, test_df, image_col, label_col, OUTPUT_DIR
    )

    # Save predictions
    result_path = save_results(predictions_df, TEST_CSV, OUTPUT_DIR, filename=RESULTS_FILENAME)

    # Validation checks
    validate_output(result_path, test_df, label_col)

    # Print model path and validation score
    print(f"Model saved to: {model_dir}")
    if val_score is not None:
        print(f"Validation AUROC: {val_score:.4f}")

    # Provide the prediction function for new images
    def predict_malignancy_probabilities(folder_path):
        """
        Predict malignancy probability (0–1) for each image in the given folder.
        Returns a DataFrame with columns: image_name, label (malignancy probability)
        """
        return predict_on_folder(model_dir, folder_path, image_col='image_name', label_col='label')

    # Example usage (uncomment to use):
    # new_folder = "/path/to/new/images"
    # preds = predict_malignancy_probabilities(new_folder)
    # print(preds.head())