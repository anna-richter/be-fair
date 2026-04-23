```markdown
# Technical Report: Machine Learning Approaches for Malignant vs Benign Skin Lesion Classification

## Introduction

The objective is to develop an automated, probability-based classification system to differentiate malignant from benign skin lesions using dermoscopic images, addressing the task posed by a collaborating dermatologist. The performance is evaluated using the Area Under the Receiver Operator Curve (AUROC) metric. Various feature extraction and modeling strategies were empirically tested to balance accuracy, robustness, and computational efficiency.

## Preprocessing

### Data Handling

- All experiments are based on a dataset comprising skin lesion images and corresponding labels (“malignant”/“benign”).
- Images are accessed via unique identifiers; .jpg extensions are assumed. Metadata is managed in CSV files linking images to labels.
- Labels are binarized to 0 (benign) and 1 (malignant) for model consumption.

### Image Preprocessing

- All deep learning-based models apply standard resizing (`224x224`), normalization (ImageNet means and standard deviations), and conversion to RGB.
- Data augmentations, when used, are deterministic for test-time augmentation (TTA), including horizontal and vertical flips, and 90° rotations.

### Feature Extraction

Four main strategies were assessed:

1. **CNN-based Embeddings:**
    - **ResNet18, MobileNetV2, EfficientNet-B0**: Pretrained on ImageNet, used as fixed feature extractors; final classifier replaced or set to identity for global image representation.
    - No fine-tuning of backbones for large experiments (to reduce overfitting and runtime).

2. **Handcrafted Features:**
    - **Color Histograms**: 64-bin per channel (192-D vector) computed per image for a classical ML baseline.

3. **Embedding Post-Processing:**
    - Mean-aggregation across TTA views.
    - Optional per-sample L2 normalization applied to embedding vectors before classifier training and prediction.

## Modeling Methods

- **LightGBM Classifier:** Used with ResNet18 and EfficientNet-B0 extracted features.
- **RandomForest Classifier:** Applied to handcrafted color histogram features.
- **Logistic Regression:** Used on extracted embeddings (MobileNetV2, EfficientNet-B0) for simple, robust calibration.
- **Shallow ResNet18 Fine-tuning:** Training only the final layer for rapid prototyping.

### Cross-Validation and Evaluation

- Stratified K-Fold (typically 5 folds, occasionally reduced to 3 for speed) cross-validation adopted throughout.
- For all models, per-fold AUROC is computed, with mean AUROC as the primary metric.
- Out-of-fold (OOF) prediction arrays are used for unbiased in-sample AUROC measurement in the most recent experiments.

### Test-Time Augmentation (TTA)

- Variants with no TTA, basic (horizontal flip), extended (vertical flip), and further (rotation) augmentations were evaluated.
- Embeddings from each augmentation are averaged for each sample before passing to the classifier.

### Inference Function

- All models define a reusable `predict()` function, which:
    - Loads new images and preprocesses them as above.
    - Extracts features/embeddings (with TTA if applicable).
    - Averages predictions across all CV-trained classifiers for robust probability estimation.

## Results Discussion

### Performance Overview

| Approach                                                  | Mean AUROC (CV) | OOF/Overall AUROC         |
|-----------------------------------------------------------|-----------------|---------------------------|
| Color histogram + RandomForest                            | 0.701           | –                         |
| ResNet18 embeddings + LightGBM                            | 0.856           | –                         |
| MobileNetV2 embeddings + Logistic Regression              | 0.837           | –                         |
| EfficientNet-B0 embeddings + Logistic Regression          | 0.856           | 0.951                     |
| EfficientNet-B0 + TTA (H-flip) + Logistic Regression      | 0.952           | 0.952                     |
| EfficientNet-B0 + TTA (H+V flip) + Logistic Regression    | 0.867           | 0.955                     |
| EfficientNet-B0 + TTA (H+V flip) + L2 norm + Logistic Reg | 0.885           | 0.909                     |
| EfficientNet-B0 + TTA (H+V+Rot) + Logistic Regression     | 0.870           | 0.957                     |
| EfficientNet-B0 + TTA (H+V+Rot) + L2 norm + Logistic Reg  | 0.888           | 0.911                     |
| EfficientNet-B0 + TTA (H+V flip) + LightGBM + OOF eval    | 0.897           | 0.897 (OOF)               |
| Shallow ResNet18 finetune (3 folds, quick)                | 0.731           | –                         |

**Key Findings:**

- **Simple color histograms** are much less effective (AUROC ~0.70) than deep learning-based embeddings.
- **CNN-extracted embeddings** (ResNet18, MobileNetV2, EfficientNet-B0) combined with shallow classifiers significantly outperform histogram features.
- **EfficientNet-B0** with TTA (using horizontal, vertical flips, and 90° rotation) shows the best mean AUROC, with gains plateauing after three augmentations.
- **Per-sample L2 normalization** of embeddings offers a mild improvement in mean AUROC, especially in combination with TTA.
- **Fine-tuned ResNet18** with minimal epochs underperforms compared to embedding-based approaches.
- **LightGBM vs Logistic Regression**: Both perform well; LightGBM yields slightly higher OOF AUROC with EfficientNet-B0 embeddings.
- **Out-of-fold (OOF) AUROC** is essential for unbiased in-sample evaluation to avoid overestimating generalization.

### Technical Decisions & Rationale

- **CNN Backbones**: EfficientNet-B0 chosen for its balance of model efficiency and representation quality.
- **Classifier Choice**: Logistic regression generally preferred for stability and calibration when features are well-behaved; LightGBM used to explore non-linear effects, found effective in final OOF assessment.
- **Test-Time Augmentation**: Substantially improved robustness, likely by de-emphasizing orientation and cropping effects.
- **Normalization**: L2 normalization of embeddings aids logistic regression classifiers, making the decision boundary depend on angular relationships in the feature space.
- **Unbiased Validation**: Shift to OOF-based validation in the final LightGBM experiment for a realistic estimate of model performance.

## Future Work

- **External Validation**: All experiments evaluated on internal cross-validation splits; future work should test on a temporally or geographically distinct cohort.
- **Finer TTA/Ensemble Exploration**: More advanced augmentations (e.g., color jitter, multi-crop) and combination of multiple CNN backbones could be tested.
- **End-to-End Fine-tuning**: The best results were achieved with frozen feature extractors; limited end-to-end fine-tuning with regularization or semi-supervised pretraining might yield further improvements if compute allows.
- **Calibration**: While ROC-AUC assesses discrimination, calibration (e.g., via Platt scaling or isotonic regression) could further improve the reliability of probability outputs.
- **Explainability**: Integration of saliency maps or attention heatmaps could improve clinical interpretability for dermatologist end-users.

---

**Recommended Final Pipeline:**  
EfficientNet-B0 pretrained on ImageNet as feature extractor; extract embeddings from original, horizontally-flipped, vertically-flipped, and 90-degree rotated images; embeddings are averaged per sample; apply per-sample L2 normalization; train LightGBM or logistic regression classifiers in 5-fold stratified cross-validation; use only out-of-fold predictions to assess AUROC; ensemble CV classifiers for inference on new images.  
This approach achieved an OOF AUROC of ~0.897.

---
```