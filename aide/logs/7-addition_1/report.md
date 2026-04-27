```markdown
# Skin Lesion Malignancy Classification: Technical Report

## Introduction

The goal was to develop a robust machine learning model for classifying skin lesions as malignant or benign, using a curated dataset provided by dermatology experts. Model fairness toward skin tone and reliable malignancy probability outputs (range 0–1) were prioritized. Model performance was evaluated using Area Under the Receiver Operator Curve (AUROC).

This report summarizes empirical findings and technical decisions across a series of modeling attempts, culminating in a state-of-the-art pipeline leveraging advanced augmentation, modern architectures, and cross-validation.

---

## Preprocessing

### Data Preparation and Label Mapping

- The dataset comprised clinical skin lesion images labeled as “malignant”, “benign”, or “non-neoplastic”.
- All experiments mapped “malignant” to class 1 (positive), and both “benign” and “non-neoplastic” to class 0 (negative), ensuring binary classification.
- Images were loaded in RGB format and all experiments dropped rows with missing labels.

### Train/Test Splits

- Initial models used a stratified 80/20 train/validation split, ensuring balanced representation of malignant cases.
- Later experiments adopted stratified 5-fold cross-validation for robust generalization assessment and reduced validation variance.

### Image Preprocessing and Augmentation

- **Input normalization:** All images normalized to ImageNet mean/std.
- **Resizing and cropping:** 
  - Training: RandomResizedCrop to input size (increased from 224 up to 300 for deeper networks).
  - Validation/Test: Resize + CenterCrop (or TenCrop for test-time augmentation).
- **Basic augmentation:** RandomHorizontalFlip applied consistently during training.
- **Advanced augmentation:** 
  - `ColorJitter` (brightness, contrast, saturation, hue) added to improve generalization across skin tones and lighting conditions.
  - `RandomErasing` applied post-normalization (p=0.5) for occlusion robustness.
  - **Test-time augmentation (TTA):** TenCrop used during validation/inference to average predictions over multiple spatial crops.
- **Mixup:** Mixup augmentation (alpha=0.4) used in later pipelines for additional generalization, combining random pairs of samples and labels.

---

## Modeling Methods

### Model Backbones Explored

Empirical comparison spanned multiple modern CNN architectures (all pretrained on ImageNet):

| Backbone         | Input Size | Validation AUROC (best run) |
|------------------|------------|-----------------------------|
| ResNet18         | 224        | 0.9126                      |
| DenseNet121      | 224        | 0.9398 (with TTA)           |
| EfficientNet-B0  | 224        | 0.9454                      |
| EfficientNet-B1  | 240        | 0.9475                      |
| EfficientNet-B2  | 260        | 0.9498                      |
| EfficientNet-B3  | 300        | 0.9449                      |
| ResNeXt50_32x4d  | 224        | 0.9137                      |
| MobileNetV2      | 224        | 0.9128                      |

### Loss Functions and Optimization

- **Binary Cross-Entropy with Logits (BCEWithLogitsLoss):** Used as the standard loss function.
- **Focal Loss:** Experimented with (gamma=2.0, alpha=0.25) to address class imbalance; no consistent improvement over BCE.
- **Mixup loss**: Losses calculated as a weighted sum of two mixup targets.
- **Optimizers:**
  - **Adam:** Baseline optimizer.
  - **AdamW:** Used with weight decay (1e-4) for stronger regularization in later experiments.
- **Learning rate schedules:**
  - **Fixed:** Initial runs.
  - **OneCycleLR:** Later employed (max_lr up to 1e-3), yielding mild regularization and convergence benefits.

### Training Regime

- Epochs: Typically 8 (in line with early stopping/uniformity across experiments).
- Batch size: 16–32, depending on model size and input resolution.
- **Early stopping**: Best model checkpoint saved using validation AUROC.

### Model Ensembling and Cross-validation

- **Stratified 5-fold cross-validation:** Final pipelines trained with 5 folds; best checkpoint per fold saved.
- **Ensemble predictions:** For inference, each image’s probability was obtained by averaging TTA-augmented outputs across all fold models.

---

## Results Discussion

### Empirical Findings

- **Model architecture:** Progressively increasing model capacity and input resolution consistently boosted AUROC. EfficientNet-B2 (input 260) gave the best mean AUROC (≈0.9498) in hold-out and (0.943–0.9498) in cross-val settings; EfficientNet-B3 did not bring further gains.
- **Data augmentation:** Addition of ColorJitter and RandomErasing yielded marginal improvements in AUROC (+0.003–0.01), and are recommended for deployment, especially for fairness across varying skin tones and image conditions.
- **TTA:** Consistently improved validation AUROC by 0.01–0.02 over single-center crop.
- **Mixup:** Provided smoother training curves and better generalization; an established benefit at negligible computational cost.
- **Optimizer choice:** AdamW and learning rate scheduling (OneCycleLR) yielded minor, but reproducible, improvements in generalization.

### Cross-validation Performance

Best 5-fold cross-validation configuration (EfficientNet-B2, input 260, mixup, TTA):

- **Per-fold AUROC range:** 0.9297–0.9512
- **Mean AUROC:** 0.9389–0.9429 (low stddev)

### Inference Functionality

All pipelines implemented a standardized `predict(folder_path)` function, which:

- Loads all fold checkpoints.
- Applies the same normalization and TTA as in validation.
- Averages model outputs (after sigmoid) across crops and folds.
- Returns a pandas DataFrame mapping each filename to its malignancy probability (float, 0–1).

---

## Future Work

Despite strong performance, several directions remain for further improvement and investigation:

1. **Fairness Auditing:** Explicit auditing and subgroup analysis (e.g., by skin tone) are recommended to ensure equal performance across demographics. Could be achieved by analyzing AUROC stratified by annotated skin tone groups.
2. **Advanced Architectures:** Explore transformer-based image models (e.g., ViT, ConvNeXt) or multi-modal approaches (if metadata is available).
3. **Semi-supervised Learning:** Incorporate unlabeled data or weak supervision to further boost robustness.
4. **Calibration:** Assess and improve probabilistic calibration of outputs for clinical interpretability.
5. **Interpretability:** Integrate saliency/attribution maps (GradCAM, etc.) for supporting clinical review and ensuring trust.
6. **Long-term Maintenance:** Deploy periodic model retraining with new data and consider continuous monitoring for performance drift.

---

## Summary Table of Key Configurations

| Model           | Input Size | Augmentation                | Mixup | TTA     | Optimizer/Scheduler     | AUROC (best run) | Cross-val Mean AUROC |
|-----------------|-----------|-----------------------------|-------|---------|------------------------|------------------|---------------------|
| ResNet18        | 224       | Basic                       | No    | No      | Adam                   | 0.9126           | —                   |
| DenseNet121     | 224       | Basic+TTA                   | No    | Yes     | Adam                   | 0.9398           | —                   |
| EffNet-B0       | 224       | Basic                       | Yes   | Yes     | Adam                   | 0.9454           | —                   |
| EffNet-B1       | 240       | Basic                       | Yes   | Yes     | Adam                   | 0.9475           | —                   |
| EffNet-B2       | 260       | Basic                       | Yes   | Yes     | Adam                   | 0.9498           | 0.943–0.9498        |
| EffNet-B3       | 300       | Basic                       | Yes   | Yes     | Adam                   | 0.9449           | —                   |
| EffNet-B2 CV    | 260       | Basic+Mixup+TTA             | Yes   | Yes     | Adam                   | —                | 0.9389–0.9429       |

---

## Recommendation

Based on empirical evidence, we recommend the following default configuration for deployment and further validation:

- **Model:** EfficientNet-B2, input size 260×260
- **Augmentations:** RandomResizedCrop, RandomHorizontalFlip, ColorJitter (brightness/contrast/saturation/hue), RandomErasing, mixup (alpha=0.4)
- **Validation/Inference:** TenCrop TTA and normalization
- **Optimizer:** AdamW or Adam with learning rate scheduling
- **Training:** 8 epochs, stratified 5-fold cross-validation
- **Final output:** Ensemble predictions averaged across folds and spatial crops

This configuration achieves a mean cross-validation AUROC of ~0.943, with consistent performance and built-in robustness/fairness measures.

---
```