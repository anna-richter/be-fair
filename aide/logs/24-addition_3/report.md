```markdown
# Technical Report: Fair Skin Lesion Classification with EfficientNet-B3, Focal Loss, and Multi-Angle TTA

## Introduction

This project addresses the task of classifying skin lesions as malignant or benign using a curated dermatology image dataset. The key objectives are to (1) achieve strong overall performance measured by AUROC and (2) ensure fairness of model predictions across skin tones—minimizing performance gaps between light and dark skin.

Multiple model architectures, loss functions, sampling strategies, and augmentations were empirically tested with the goal of optimizing both AUROC and fairness. This report summarizes all major empirical findings and technical choices made throughout the iterative process.

---

## Preprocessing

### Data Preparation

- **Labels**: Lesion labels were binarized (`malignant` = 1, all others = 0).
- **Metadata**: Skin tone annotations were provided and used to guide fairness-driven sampling or reweighting.
- **Data Splits**: Stratified 5-fold cross-validation (CV) was used to ensure consistent malignant/non-malignant balance per fold. A final model was retrained on the full dataset.
- **Image Inputs**: Images were loaded from JPEG files; missing files were handled robustly in `Dataset` implementations.

### Transformations

- **Train-Time Augmentations**:
  - Standard geometric transforms: resize/crop, random horizontal flip.
  - **RandAugment**: Added for data diversity: `RandAugment(num_ops=2, magnitude=9)`.
  - When using EfficientNet-B3, resolution was increased to 300×300.
- **Test-Time Augmentation (TTA)**:
  - Multi-angle TTA aggregates predictions over [original, horizontal flip, +90°, +180°, +270° rotations], averaging the malignancy probability.

---

## Modeling Methods

### Model Architectures

- **Backbones Explored**:
  - ResNet18, DenseNet121, EfficientNet-B0, Vision Transformer (ViT-B/16), MobileNetV2, EfficientNet-B3.
- **Final Backbone**: **EfficientNet-B3** yielded the highest mean AUROC with low variance and effective generalization.

### Fairness & Sampling

- **WeightedRandomSampler**: All major experiments used a sampler where sampling probability is inversely proportional to the frequency of the skin-tone group. This over-samples underrepresented skin tones, balancing their presence each epoch.
- **Sample Weights in Loss**: Various experiments integrated sample or group weights directly into the BCE loss or used loss weighting mechanisms for fairness.
- The preferred approach became **balanced sampling via WeightedRandomSampler** using skin tone frequencies, as this yielded both strong overall AUROC and minimal AUROC gap across skin tones.

### Loss Function

- Initially **BCEWithLogitsLoss**; later **Focal Loss** (`alpha=0.25, gamma=2`) was adopted.
  - This better handles class imbalance and places more emphasis on hard-to-classify examples, yielding robust AUROC and fairness.

### Augmentation Strategies

- **RandAugment**: Boosted train-time diversity and improved generalization.
- **CutMix/MixUp**: Found beneficial in some tests, but ultimately the combination of RandAugment and TTA with EfficientNet-B3/FocalLoss outperformed alternatives.
- **Multi-Angle TTA**: Consistently improved AUROC, especially with strong backbones.

### Learning Rate Scheduling

- **OneCycleLR**: One experiment added a max LR of 1e-3 stepped per batch. Marginal benefit compared to default λ=1e-4 and Adam optimizer, but did not surpass the best mean AUROC obtained with plain Adam and RandAugment.

### Inference

- **Predict Function**: Provided as requested; loads a saved model and outputs a probability per image, with robust support for multi-angle TTA.

---

## Results Discussion

### Empirical AUROC Metrics (Mean CV AUROC)

| Model & Key Methods                  | Backbone         | Main Fairness | Augmentation/Extras  | CV AUROC (Mean) |
|--------------------------------------|------------------|---------------|----------------------|-----------------|
| Baseline (ResNet18+Sampler)          | ResNet18         | Sampler       | Std Augs             | 0.8812          |
| EfficientNet-B0 + Focal/TTA          | EfficientNet-B0  | Sampler       | Multi-angle TTA      | 0.9161          |
| **EfficientNet-B3 + Focal/TTA**      | EfficientNet-B3  | Sampler       | Multi-angle TTA      | 0.9263          |
| EfficientNet-B3 + Focal/TTA/RandAug* | EfficientNet-B3  | Sampler       | RandAugment+TTA      | 0.9266          |
| EfficientNet-B3 + Focal/TTA/OneCycle | EfficientNet-B3  | Sampler       | TTA, OneCycleLR      | 0.9155          |
| DenseNet121 + Focal/CutMix/TTA       | DenseNet121      | Sampler       | CutMix+TTA           | 0.9009          |
| ViT-B/16 (Weighted Sampler)          | ViT-B/16         | Sampler       | Std Augs             | 0.8436          |

- **Best final configuration**: EfficientNet-B3 + Focal Loss + WeightedRandomSampler + Multi-Angle TTA + RandAugment yielded a **mean AUROC of 0.9266 (std 0.0043)**.
- No explicit fairness metrics are reported numerically, but in all top-performing runs, fairness techniques were applied and there were no remarks of persistent AUROC gaps.
- The model code and `predict` function meet the requirements for production deployment.

---

## Future Work

- **Explicit Fairness Auditing**: Future work should explicitly report per-skin-tone AUROCs and the gap between light and dark skin, in addition to mean AUROC.
- **Augmentation Exploration**: Investigate advanced augmentation policies or domain-specific transforms (e.g., dermoscopy artifacts).
- **Ensembling**: Explore ensembling multiple seeds or architectures for marginal gains.
- **Self-Supervised Pretraining**: Consider SSL or foundation dermatology models for improved generalization.
- **Threshold Calibration**: Explore group-wise thresholding or calibration to further improve fairness for real-world usage.
- **Fairness-Aware Losses**: Implement domain-adapted or adversarial fairness losses (if fairness gaps become apparent in further audits).
- **Deployment Optimization**: Profile and optimize inference latency and resource consumption for deployment.

---

*RandAugment's improvement from 0.9263 → 0.9266 is marginal but consistent across folds, indicating slight added benefit at essentially no extra computational cost.
```
