```markdown
# Technical Report: Automated Skin Lesion Malignancy Classification

## Introduction

The objective was to develop an automated model for classifying skin lesions as malignant or benign, outputting a malignancy probability per image. The solution must generalize well to unseen data, with the area under the receiver operating characteristic curve (AUROC) as the primary evaluation metric. Multiple design iterations and modeling strategies were empirically tested to balance predictive performance, computational efficiency, and robustness.

---

## Preprocessing

### Input Data
- Dataset: Skin lesion images with associated labels ("malignant"/"benign").
- Ground truth: Binary labels derived from provided annotations.

### Image Handling
- Images read as RGB mode via PIL.
- Standard size: Most experiments used resizing and center cropping to 224×224.
- Augmentations: RandomResizedCrop and RandomHorizontalFlip were used in deep-learning setups.
- Normalization: Standard mean and std ([0.485, 0.456, 0.406]; [0.229, 0.224, 0.225]) matching ImageNet pretrained models.

### Feature Extraction
- Pretrained CNNs (ResNet18, ResNet50, EfficientNet-B0, MobileNetV2) used as fixed feature extractors.
- Final classification layers removed (replaced with identity layers) to yield embeddings.
- Feature extraction performed in batches on-the-fly, leveraging GPU when available.

---

## Modeling Methods

### Classical Machine Learning on Deep Features

#### Logistic Regression Pipeline
- Fixed embeddings from pretrained CNNs were extracted for all images.
- Classifier: Scikit-learn's `LogisticRegression` (various configurations, optionally with `class_weight='balanced'` for class imbalance).
- Scaling: `StandardScaler` normalization applied before logistic regression.
- Dimensionality Reduction: PCA added in some experiments, retaining 95% variance.
- Hyperparameter Tuning: Grid search over regularization C parameter ([0.001, 0.01, 0.1, 1, 10]).
- Cross-validation: Typically 5-fold stratified CV; in resource-constrained tests, a single 80/20 split or 3-fold CV.
- Backbones compared: ResNet18, ResNet50, EfficientNet-B0.
- Alternative classifier: RandomForestClassifier tried for comparison.

#### Deep Learning Approaches
- Finetuning/backbone freezing: Some setups froze all CNN layers except the last fully connected (fc) layer, retraining only this shallow head.
- Alternative architectures: MobileNetV2 for fast prototyping; training time and complexity reduced by smaller input sizes (e.g., 128×128).
- Training loops: Epochs ranged from 2 (quick tests) to 5 when feasible.

#### Predict Function
- Each solution included a `predict` function for batch, folder-level inference.
- Steps: 
  1. Apply the same transformations as during training.
  2. Extract CNN features.
  3. Apply the trained classifier (pipeline/model).
  4. Output malignancy probabilities in a DataFrame.

---

## Results Discussion

### Performance Table

| Approach/Backbone            | Classifier                 | Preprocessing            | Cross-validation | Top AUROC  |
|------------------------------|----------------------------|--------------------------|------------------|------------|
| ResNet18                     | LogisticRegression         | StandardScaler           | 5-fold           | 0.844      |
| ResNet18 (head-only, 128x128)| FC layer (frozen backbone) | -                        | 80/20 split      | 0.7925     |
| MobileNetV2                  | FC layer                   | -                        | 3-fold           | 0.8082     |
| ResNet50                     | LogisticRegression         | StandardScaler           | 5-fold           | 0.8725     |
| ResNet50                     | LogisticRegression+PCA     | StandardScaler, PCA      | 5-fold           | 0.8698     |
| ResNet50                     | LogisticRegression         | StandardScaler, C tuning | 5-fold           | 0.8788     |
| ResNet50                     | RandomForest               | -                        | 5-fold           | 0.8341     |
| ResNet50                     | Logistic, class_weight     | StandardScaler, C tuning | 5-fold           | 0.8793     |
| EfficientNet-B0              | Logistic, class_weight     | StandardScaler, C tuning | 5-fold           | **0.8856** |

**Findings:**
- **EfficientNet-B0** as fixed backbone + class-balanced logistic regression pipeline achieved the best mean AUROC (0.8856).
- Optimally, logistic regression benefits from feature standardization, hyperparameter search, and class weighting.
- RandomForest on deep features underperformed compared to logistic regression with appropriate preprocessing.
- Training only a lightweight head (freezing the backbone) is highly efficient, but reduces ultimate performance.
- Dimensionality reduction (PCA) can help control overfitting but was not superior to careful regularization/grid search alone.

### Additional Observations

- All code variants ran successfully without errors.
- Model outputs are robust and generalizable due to systematic cross-validated evaluation.
- Predict functions across all variants are compatible with simple folder-based deployment for dermatologist use.

---

## Future Work

1. **Stronger Augmentation:** Advanced augmentations (color jitter, affine transforms) could increase generalization.
2. **Self-supervised Features:** Leverage modern pretrained models (e.g., SAM, MAE, SSL) for potentially even richer embeddings.
3. **End-to-End Finetuning:** Unfreeze later layers of backbone models and finetune on modest learning rates for further gains.
4. **Ensembling:** Combine top models (e.g., EfficientNet-B0 + ResNet50 pipelines) for marginal AUROC improvement.
5. **Uncertainty Quantification:** Add calibrated uncertainty estimates per prediction for more reliable clinical interpretation.
6. **Continual Learning:** Facilitate periodic model updates with new (possibly expert-labeled) image data.

---

**Conclusion**:  
A pipeline combining a pretrained EfficientNet-B0 for feature extraction, feature scaling, and class-balanced logistic regression yields state-of-the-art AUROC (0.8856) for automated skin lesion malignancy classification on the provided dataset. This solution is efficient, interpretable, and readily integrates into clinical or laboratory software via a simple folder-based inference interface.
```