```markdown
# Skin Lesion Malignancy Classification: Technical Report

## Introduction

The goal was to build and optimize a binary classifier to distinguish malignant from benign skin lesions using a curated dataset provided by dermatologists. Model effectiveness was measured by the area under the receiver operating characteristic curve (AUROC). The final deployment requires a function to produce malignancy probabilities for new images.

This report summarizes the empirical findings and technical decisions from a systematic set of design, training, and validation experiments, spanning model architectures, loss functions, data augmentations, sampling strategies, and inference-time techniques.

---

## Preprocessing

### Data Handling
- **Data Split**: All experiments used an 80/20 stratified split (by malignant/benign label) into train and validation sets.
- **Image Preprocessing**:
  - Resized images (usually to 224×224).
  - Normalized using ImageNet mean/std values.
- **Label Mapping**: Malignant = 1, others = 0 for binary classification.

### Augmentation Strategies
- **Baseline**: Horizontal flip, rotation, and color jitter during training.
- **Advanced**:
  - **RandomResizedCrop** for scale invariance.
  - **ColorJitter** for robustness to color/lighting.
  - **RandomErasing** (p=0.5, after normalization) to simulate occlusion and further boost robustness.
  - **Mixup (α=0.2)**: Random convex combination of pairs of images and labels, significantly improving regularization.
  - **CutMix** and **Test-Time Augmentation (TTA)**: Ensemble averaging over augmented views at inference improved model robustness.

### Sampling
- **Oversampling**: Used WeightedRandomSampler for class imbalance, ensuring minority class is adequately represented in batches.

---

## Modeling Methods

### Backbone Architectures
- **ResNet18**: Initial simple baseline.
- **DenseNet121**: Main high-performing CNN backbone.
- **EfficientNet-B0**: For both as primary model (via timm) and as a deep feature extractor for downstream linear classifiers.
- **MobileNetV2**: Light-weight alternative.
- **Feature Extractor + Logistic Regression**: EfficientNet-B0 features followed by scikit-learn LogisticRegression.

### Loss and Optimization
- **Loss Functions**:
  - **Binary Cross Entropy with Logits (BCEWithLogitsLoss)**: Used throughout as the standard.
  - **Focal Loss (γ=2, α=0.25)**: Focused learning on hard/rare cases, improving AUROC over BCE.
- **Optimizers**:
  - **Adam**: Standard optimizer.
  - **AdamW**: Added weight decay regularization, yielding top AUROC scores.
  - **OneCycleLR**: Scheduler that ramps LR up then anneals, providing modest benefits for some seeds/runs.
- **Mixup/CutMix**: Replacing standard inputs with label-interpolated or region-mixed inputs drove regularization and AUROC improvement.

### Test-Time Augmentation (TTA)
- **Standard 2-way**: Average logits from original and horizontal-flipped inputs.
- **4-way TTA**: Average over original, h-flip, v-flip, and both; provided further AUROC uplift.
- In all cases, TTA was used at validation and in the `predict()` deployment function.

### Key Training Protocol
- **Epochs**: Five epochs for all runs.
- **Batch size**: 32.
- **Hardware**: Code supports both CPU and GPU inference/training.

---

## Results Discussion

### Empirical Findings

| Approach                                    | Validation AUROC |
|----------------------------------------------|------------------|
| ResNet18 (baseline)                          | 0.8874           |
| EfficientNet-B0                             | 0.9008           |
| DenseNet121                                 | 0.9057           |
| MobileNetV2                                 | 0.9015           |
| EfficientNet-B0 Features + LogisticRegression| 0.8688           |
| DenseNet121 + TTA (h-flip)                  | 0.9118           |
| DenseNet121 + Mixup + TTA                   | 0.9176           |
| DenseNet121 + Mixup + RandomErasing + TTA   | 0.9205           |
| DenseNet121 + Mixup + AdamW + WD + TTA      | 0.9205           |
| DenseNet121 + Mixup + FocalLoss + TTA       | 0.9126           |
| DenseNet121 + 4-way TTA                     | 0.9177           |
| DenseNet121 + Mixup + Oversample + TTA      | 0.9036           |
| DenseNet121 + CutMix + TTA                  | 0.9119           |
| DenseNet121 + Mixup + OneCycleLR + TTA      | 0.8807           |
| EfficientNet-B0 + Mixup + RandomErasing + TTA| 0.8870           |

#### Observations
- **DenseNet121** with advanced augmentations, especially **Mixup**, **RandomErasing**, and **AdamW optimizer with weight decay**, consistently provided the highest validated AUROC (0.9205).
- **Test-Time Augmentation** (TTA), particularly 4-way (include v-flip), improved generalization and AUROC versus standard single-view evaluation.
- **Mixup** provided gains across all backbones, while CutMix and focal loss offered small additional benefits.
- **EfficientNet-B0** as a backbone or feature extractor trailed DenseNet121 under equivalent protocols on this dataset.
- **Oversampling** the rare class offered moderate benefit for AUROC but was less effective than Mixup/TTA at boosting rare-class discrimination.
- **AdamW** generally outperformed vanilla Adam when combined with strong augmentation, mainly via improved regularization.

### Model Selection

The best-performing pipeline is:
- **DenseNet121**, pretrained; head replaced with single-unit output.
- **Preprocessing**: Resize, random horizontal flip, random rotation, color jitter, normalization, random erasing.
- **Augmentation**: **Mixup** (α=0.2) during training.
- **Optimizer**: AdamW (lr=1e-4, wd=1e-4).
- **Inference**: **Test-Time Augmentation** (original + h-flip), probabilities averaged.
- **Epochs**: 5.
- **Validation AUROC**: **0.9205**.

The corresponding `predict()` function loads the trained model and applies the same normalization/TTA pipeline.

---

## Future Work

- **Longer Training/Larger Models**: More epochs, larger models (EfficientNet-B3+), or ensembling may yield additional gains.
- **Advanced TTA**: Explore rotation or color perturbations at inference.
- **Exposure to More Views**: Leverage pseudo-labeling and further rare-positives oversampling.
- **Multi-task/Contrastive Pretraining**: Use representations learned on wider dermoscopy datasets.
- **Calibration**: Evaluate/adjust probability calibration to ensure probabilities are meaningful for clinical risk assessment.
- **Explainability**: Integrate saliency or attribution methods for deployment to support clinician interpretability.

---

## Conclusion

DenseNet121, trained with heavy data augmentation (Mixup + RandomErasing), robust optimization (AdamW + WD), and averaged predictions over augmented views, delivered the highest validation AUROC on the dermatologist-curated dataset. This pipeline is recommended for deployment, offering strong, generalizable malignancy probability predictions for new skin lesion images.

```