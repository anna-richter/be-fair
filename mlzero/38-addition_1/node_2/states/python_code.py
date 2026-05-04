"""
Skin Lesion Malignancy Prediction using AutoGluon Tabular

This script trains a tabular predictor to classify skin lesions as malignant or benign using only tabular metadata.
It:
- Loads and preprocesses the data (drops NA labels, removes index column if present).
- Trains an AutoGluon TabularPredictor with presets="extreme" and eval_metric="roc_auc".
- Saves the trained model to a timestamped folder in the specified output directory.
- Predicts malignancy probability for each test sample, preserving original indices and output format.
- Provides a function to predict on a new folder of images (using only tabular features, not image content).
- Performs validation checks on the output.
- Prints AUROC on the internal validation set.

# Installation (if needed):
# !pip install autogluon.tabular pandas

Author: AutoML Agent
"""

import os
import random
import time
import pandas as pd
import numpy as np

from autogluon.tabular import TabularPredictor

# Paths
DATA_DIR = "/home/anri21/be-fair/mlzero/addition_1_data"
OUTPUT_DIR = "/home/anri21/be-fair/mlzero/38-addition_1/node_2/output"
TRAIN_CSV = os.path.join(DATA_DIR, "mydataset.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_timestamp_folder(base_dir):
    """Generate a random timestamped folder path under base_dir."""
    timestamp = int(time.time()) + random.randint(0, 99999)
    folder = os.path.join(base_dir, f"model_{timestamp}")
    return folder

def map_label_to_binary(label):
    """Map label to binary: malignant=1, else=0."""
    return 1 if str(label).strip().lower() == "malignant" else 0

def main():
    # 1. Data Loading and Preprocessing
    df = pd.read_csv(TRAIN_CSV)
    # Remove unnecessary index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    # Remove training samples without valid labels (drop NA in 'label')
    df = df.dropna(subset=['label'])
    # Map label to binary (malignant=1, else=0)
    df['label'] = df['label'].apply(map_label_to_binary)
    # Remove columns that are not usable for tabular prediction (image_name)
    # (If you want to use image_name as a categorical feature, comment out the next line)
    if 'image_name' in df.columns:
        df = df.drop(columns=['image_name'])

    # 2. Model Training
    model_dir = get_timestamp_folder(OUTPUT_DIR)
    os.makedirs(model_dir, exist_ok=True)
    predictor = TabularPredictor(
        label='label',
        problem_type='binary',
        eval_metric='roc_auc',
        path=model_dir
    )
    # Use all data for training; AutoGluon handles internal validation
    predictor.fit(
        train_data=df,
        presets="extreme",
        time_limit=None,
        verbosity=2
    )
    # Model is already saved by AutoGluon

    # 3. Prediction on Test Set
    # For this script, we assume test data is a CSV file named "test.csv" in DATA_DIR
    TEST_CSV = os.path.join(DATA_DIR, "test.csv")
    if os.path.isfile(TEST_CSV):
        test_df = pd.read_csv(TEST_CSV)
        # Remove unnecessary index column if present
        if 'Unnamed: 0' in test_df.columns:
            test_df = test_df.drop(columns=['Unnamed: 0'])
        # Remove image_name if not used
        if 'image_name' in test_df.columns and 'image_name' not in df.columns:
            test_df = test_df.drop(columns=['image_name'])
        # Save original indices for validation
        orig_indices = test_df.index.copy()
        # Predict probabilities
        preds = predictor.predict_proba(test_df)
        # For binary, predictor.predict_proba returns two columns: 0 and 1
        if 1 in preds.columns:
            malignancy_prob = preds[1].values
        else:
            malignancy_prob = preds['1'].values
        # Output DataFrame: match test input format, add malignancy_probability column
        results_df = test_df.copy()
        results_df['malignancy_probability'] = malignancy_prob
        # Save results with the same format as test input (CSV)
        results_path = os.path.join(OUTPUT_DIR, "results.csv")
        results_df.to_csv(results_path, index=False)
        # Validation checks
        # 1. Indices preserved
        assert all(results_df.index == orig_indices), "Row indices mismatch!"
        # 2. Output columns: must match test_df columns + 'malignancy_probability'
        expected_cols = list(test_df.columns) + ['malignancy_probability']
        assert list(results_df.columns) == expected_cols, "Column names mismatch!"
        # 3. Output format
        assert results_path.endswith('.csv'), "Output file format mismatch!"
        # 4. Number of predictions
        assert len(results_df) == len(test_df), "Number of predictions does not match number of test samples!"
        # 5. Sanity check: probabilities in [0,1]
        assert np.all((results_df['malignancy_probability'] >= 0) & (results_df['malignancy_probability'] <= 1)), "Probabilities out of range!"
        print(f"Predictions saved to {results_path}")

    # 4. Provide the prediction function for new images (tabular only)
    # This function expects a DataFrame with the same columns as training (excluding label)
    import dill
    def predict_on_tabular(tabular_df):
        """
        Predict malignancy probability for each row in the given DataFrame.
        Returns a DataFrame with the same index and a 'malignancy_probability' column.
        """
        preds = predictor.predict_proba(tabular_df)
        if 1 in preds.columns:
            malignancy_prob = preds[1].values
        else:
            malignancy_prob = preds['1'].values
        result_df = tabular_df.copy()
        result_df['malignancy_probability'] = malignancy_prob
        return result_df

    func_path = os.path.join(OUTPUT_DIR, "predict_on_tabular.dill")
    with open(func_path, "wb") as f:
        dill.dump(predict_on_tabular, f)
    print(f"Prediction function saved to {func_path}")

    # 5. Print AutoGluon's internal validation score
    try:
        leaderboard = predictor.leaderboard(silent=True)
        print("AutoGluon internal validation leaderboard:")
        print(leaderboard)
    except Exception as e:
        print(f"Validation failed: {e}")

if __name__ == "__main__":
    main()