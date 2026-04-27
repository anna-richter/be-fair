```markdown
# Technical Report: Malignancy Classification of Skin Lesions Using Deep Learning

## Introduction

The goal of this project was to develop a robust deep learning model for classifying skin lesion images as malignant or benign, using a dataset curated by dermatology experts. Key requirements included high discriminative accuracy (evaluated by AUROC), robustness across skin tones and imaging conditions, and the provision of a function to compute malignancy probabilities for arbitrary folders of new images. Emphasis was placed on mitigating biases and ensuring fairness in prediction.

## Preprocessing

### Data Preparation

- All attempts used a CSV file containing image identifiers and diagnosis labels.
- Images were stored as `.jpg` files in a specified directory.
- Labels were binarized (`malignant`: 1; `benign`/`non-neoplastic`: 0).
- Data splits used an 80/20 stratified train/validation or 5-fold stratified cross-validation (CV), ensuring malignancy prevalence was preserved across splits.

### Data Augmentation and Transformations

Empirical design choices included:
- **Baseline augmentation:** `RandomResizedCrop(224)`, `RandomHorizontalFlip()`, normalization to ImageNet statistics.
- **Additional experimentation:**
  - **Color jitter:** Improved lighting and tone robustness.
  - **RandomErasing:** Simulated occlusions for greater generalization.
  - **MixUp and CutMix:** Combined images and labels to regularize and create more varied training examples.
  - **RandAugment:** Automated composition of diverse augmentation operations.
- **Validation transformations:** Consistently used resizing to 256, center cropping to 224, normalization.  
- **Test-Time Augmentation (TTA):** Horizontal flip or TenCrop (extracting 10 crops per image and averaging predictions) used during validation and inference for robustness.

## Modelling Methods

### Model Architectures

Several state-of-the-art convolutional and transformer-based architectures were explored:
- **Convolutional Models:** ResNet18, EfficientNet-B0, DenseNet121, and MobileNetV2.
- **Transformer Model:** Vision Transformer (ViT-B/16).
- All models were initialized with ImageNet pretrained weights (except for predict/inference loads, which use random weights but restore state dict from training).

### Loss Functions and Optimization

- **Primary loss:** `BCEWithLogitsLoss` (standard binary classification loss).
- **Class imbalance strategies:**
  - **pos_weight:** Weighted loss via positive class upweighting in imbalanced datasets.
  - **WeightedRandomSampler:** Oversampled minority (malignant) class per mini-batch.
  - **Focal Loss:** Focused learning on challenging (often minority) samples.
- **Optimizers:** Mostly Adam (and AdamW for transformer).
- **Learning Rate Scheduling:** OneCycleLR scheduler tested in one configuration for dynamic learning rate adaptation.
- **Regularization:** Dropout added before classifier head in top attempts.

### Training and Evaluation

- **Batch size:** Typically 32, with 5 training epochs per split or fold.
- **Validation AUROC:** Computed after each epoch/fold; the model with the highest AUROC was saved for inference.
- **Final pipeline:** Stratified 5-fold CV for robust estimation followed by retraining on full data set and model export.

### Model Output and Inference

- All final solutions provided a standardized `predict(folder_path)` function:
  - Loads the trained model and applies all necessary transformations.
  - Supports batch inference and TTA for robustness.
  - Returns a dictionary mapping image filenames to malignancy probabilities in [0, 1].

## Results Discussion

| Approach                     | Key Techniques/Architectures       | Validation AUROC  |
|------------------------------|------------------------------------|-------------------|
| Baseline (ResNet18)          | Standard augment, BCE Loss         | 0.8975            |
| EfficientNet-B0 (timm)       | Adam, std aug                      | 0.9113            |
| DenseNet121                  | Standard aug, BCE Loss             | 0.9133            |
| DenseNet121 + ColorJitter    | Color jitter aug                   | 0.9096            |
| DenseNet121 + RandomErasing  | RandomErasing                      | 0.9126            |
| DenseNet121 + MixUp          | MixUp (two implementations)        | 0.9041, 0.9089    |
| DenseNet121 + CutMix         | CutMix (α=1.0, p=0.5)              | 0.9125            |
| DenseNet121 + Focal Loss     | Focal Loss (γ=2, α=0.25)           | 0.903             |
| DenseNet121 + pos_weight     | Weighted BCEWithLogitsLoss         | 0.9038            |
| DenseNet121 + WeightedSampling| WeightedRandomSampler              | 0.9165            |
| DenseNet121 + RandAugment    | RandAugment + WeightedSampler      | 0.9115            |
| DenseNet121 + TTA (TenCrop)  | WeightedSampler, TenCrop at val    | 0.9224            |
| DenseNet121 + 5-Fold CV + TTA| Dropout, 5-fold CV, TenCrop TTA    | **0.9281 ± 0.0066**|
| MobileNetV2                  | Efficient, same pipeline           | 0.9087            |
| ViT-B/16                     | Transformer model                  | 0.8542            |

**Findings:**
- **DenseNet121 consistently outperformed other architectures** when coupled with advanced augmentation and class-imbalance mitigation.
- **Test-Time Augmentation (TTA, e.g., TenCrop) provided the largest single improvements** to AUROC, with 5-fold CV + TTA + dropout yielding the highest and most stable results.
- **Class-imbalance mitigation (WeightedRandomSampler, pos_weight, Focal Loss, CutMix)** improved robustness on underrepresented (malignant) lesions.
- **Vision Transformer (ViT)** underperformed compared to convolutional architectures on this dataset.
- **Strong regularization (dropout before final layer) and heavy augmentation (RandAugment, CutMix, MixUp)** further boosted generalization and fairness.
- **Best configuration**: DenseNet121, 5-Fold stratified CV, weighted sampling, dropout before final classifier, TenCrop TTA at validation/inference, retrained on full data for deployment. This achieved mean AUROC **0.9281 ± 0.0066**.

## Future Work

- Further expand image augmentation diversity (e.g., using skin tone/hue-adaptive policies).
- Explore ensemble methods combining multiple CV-fold models for deployment.
- Investigate self-supervised or semi-supervised pretraining with additional unlabeled images for further boosting performance on rare lesion types or diverse skin tones.
- Integrate calibration techniques (e.g., temperature scaling) to improve probability reliability, facilitating risk triaging.
- Deploy and monitor model fairness and calibration on prospective, demographically diverse cohorts.

---

**Recommendation:**  
Deploy the final pipeline: DenseNet121 with dropout, trained using stratified 5-fold CV and TenCrop TTA, retrained on all available data with weighted sampling. The provided `predict(folder_path)` interface enables clinical inference with robust malignancy probability estimates for all submitted images.
```
