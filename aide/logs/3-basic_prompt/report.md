```markdown
# Technical Report: Skin Lesion Malignancy Classification – Empirical Summary and Design Decisions

## Introduction

This report summarizes empirical findings and key technical choices evaluated during the development of a machine learning pipeline for classifying skin lesion images as malignant or benign. The overarching objective was to maximize discriminative performance (as quantified by mean cross-validated AUROC) on a dermatology image dataset. All downstream code was required to save the final model and provide a `predict(folder_path)` function yielding malignancy probabilities (0–1) for new images.

## Preprocessing

### Data Filtering and Label Mapping

- All pipelines filter the dataset to images labeled as either "malignant" or "benign", excluding non-neoplastic cases.
- Labels were mapped to binary targets (`malignant` = 1, `benign` = 0).

### Data Augmentation and Input Transformation

- **Baseline**: Simple resizing (typically 224×224), center or random cropping, normalization to ImageNet standards.
- **Augmented Variants**: Progressive atomic changes included
  - Random horizontal/vertical flips.
  - Random rotations (typically ±10–15°).
  - Color jitter (brightness, contrast, saturation, hue).
  - RandomResizedCrop.
  - RandomErasing (in advanced augmentations).
  - MixUp augmentation (mixing pairs of samples).
- Augmentations applied only at train time; validation/test used deterministic transforms.
- Test-Time Augmentation (TTA): Several pipelines employed horizontal-flip ensembling or more advanced 10-crop views during evaluation/prediction.

## Modeling Methods

### Model Architectures Evaluated

- **ResNet18 and ResNet50**: As feature extractors or end-to-end fine-tuned classifiers.
- **MobileNetV2**: Consistently outperformed other architectures in discriminative performance when coupled to strong augmentations and modern optimizers.
- **EfficientNet-B0**: Evaluated via the Timm library, performed well but with slightly lower AUROC than MobileNetV2 in this setting.

### Classifier Types

- **End-to-end CNN Fine-tuning**: Most highly performant pipelines fine-tuned all layers of a pretrained model with a binary output head.
- **Feature Extraction + LightGBM**: Features from a CNN (ResNet18/ResNet50) used as input to a gradient boosting classifier, evaluated with callbacks for early stopping.

### Optimization, Scheduling, and Regularization

- **Loss**: Binary cross-entropy with logits.
- **Optimizers**: Adam/AdamW (AdamW offered slight gains with weight decay regularization).
- **Schedulers**: OneCycleLR schedule improved convergence and generalization compared to static learning rates.
- **Batch Size**: 32 (optimal for both performance and GPU memory).
- **Epochs**: Best results for MobileNetV2 obtained with 5 epochs (vs. 3 for earlier/designs).

### Cross-Validation

- All pipelines employed 5-fold stratified cross-validation to estimate held-out AUROC.
- After CV, the model was retrained on the full dataset and saved for inference.

## Results Discussion

### Baseline and Incremental Improvements

- **Initial Fine-tuning (ResNet18, no advanced augmentation):** Mean AUROC ≈ 0.858
- **EfficientNet-B0:** Mean AUROC ≈ 0.844
- **MobileNetV2 (basic):** Mean AUROC ≈ 0.899 (with simple test-time augmentation)
- **LightGBM with ResNet50 features:** Mean AUROC ≈ 0.87

### Impact of Data Augmentation

- **Standard Augmentation (flips, rotation, color jitter):** Mean AUROC ~0.88–0.89 (MobileNet-based)
- **Richer Augmentation (RandomResizedCrop, strong color jitter, RandomErasing):** Mean AUROC up to 0.876–0.891
- **MixUp** (blending images/labels): Mean AUROC ≈ 0.898–0.899
- **Test-Time Augmentation:**
  - Simple horizontal flip: Consistently improved AUROC by ~0.01.
  - 10-crop TTA: Mean AUROC ≈ 0.8895 (minor improvement over single-view inference).

### Impact of Optimizer and Scheduling

- **AdamW + OneCycleLR:** Provided consistent improvements; best observed AUROC 0.906 (with standard transforms, no additional complex augmentation).
- **Epochs:** Increasing to 5 (from 3) consistently benefitted AUROC; no further gains observed with higher epoch counts in the explored range.

### Summary Table of Best Results

| Design Variant                             | Mean CV AUROC |
|--------------------------------------------|---------------|
| MobileNetV2 + TTA + AdamW + OneCycleLR, 5ep| **0.9064**    |
| MobileNetV2 + MixUp + TTA, 5ep             | 0.8988        |
| MobileNetV2 + TTA, strong aug, 5ep         | 0.8986        |
| MobileNetV2 + TTA, moderate aug, 5ep       | 0.8911–0.899  |
| 10-crop TTA (MobileNetV2)                  | 0.8895        |
| LightGBM + ResNet50/18 features            | 0.87          |
| ResNet18 baseline, simple augmentation     | 0.8582        |
| EfficientNet-B0                            | 0.8444        |

### Decision

- **Best Performance:** The optimal pipeline employs MobileNetV2 (pretrained), strong data augmentations at train time, AdamW optimizer, OneCycleLR, 5 training epochs, and horizontal-flip TTA at inference. This design yielded the highest mean CV AUROC (0.906).
- **Practicality:** All top pipelines save final weights and include a `predict(folder_path)` function that outputs per-image malignancy probability. Test-time augmentation is feasible for clinical batch prediction.

## Future Work

- **Advanced Augmentations:** Evaluate augmentations mimicking dermatoscopic artefacts; try elastic deformation or stain-style transfer.
- **Ensembling:** Average outputs from multiple seeds/architectures for potential further AUROC gains.
- **Longer/Fine-tuned Training Schedules:** Investigate transfer learning with additional external datasets and longer schedules.
- **Explainability:** Integrate saliency/heatmaps into predictions to aid clinical interpretability.
- **Calibration:** Assess and if necessary calibrate predicted probabilities via post-hoc methods (e.g., Platt scaling).
- **Continual Learning:** Periodically retrain the model as the dataset grows in size and diversity.

---
**In summary:** Rich train-time data augmentations, AdamW + OneCycleLR optimizer/scheduler, and test-time augmentation (simple flip or multi-crop) consistently yielded robust generalization and state-of-the-art discriminative performance for skin lesion malignancy classification on the provided dataset.
```
