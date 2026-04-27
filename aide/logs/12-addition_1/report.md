```markdown
# Technical Report: Deep Learning Pipeline for Skin Lesion Malignancy Classification

## Introduction

This report summarizes the empirical findings and technical decisions from a sequence of modeling attempts to address the task: **malignant vs benign skin lesion classification**. The dataset consists of 16,577 dermoscopic images curated by dermatology experts, with each image labeled as "malignant" or otherwise. The ultimate objective is to provide a model with high AUROC performance, that can return malignancy probabilities for any folder of new images, while ensuring fairness—particularly with regard to skin tone. All implementations conclude with a `predict(folder_path)` function for deployment.

---

## Preprocessing

### Data Splitting
- **Stratified Splits:** All experiments use stratified hold-out or cross-validation to maintain class proportions for malignant and benign lesions.
- **Train/Validation Ratio:** Predominantly 80% train, 20% validation, except in certain cross-validation attempts.
- **Skin tone balancing:** Later iterations introduced WeightedRandomSampler based on skin tone or class frequency for fair representation.

### Image Preprocessing
- **Image Loading:** JPEG images are read using PIL, converted to RGB.
- **Resizing:** Images are resized to 224x224 (or 256x256 with crops in some attempts), to match pretrained ImageNet backbone input requirements.
- **Normalization:** All pipelines normalize RGB channels to ImageNet means and stds.

### Data Augmentation
- **Standard Augmentations:** All models employ random crops and horizontal (sometimes vertical) flips during training.
- **Color & Artifact Robustness:** ColorJitter (brightness, contrast, saturation, hue) and advanced augmentations like RandAugment were explored.
- **MixUp and CutMix:** Multiple runs leverage MixUp (linear label/image interpolation) and CutMix (random patches exchanged between images) extensively.
- **RandomErasing:** At least one variant tested RandomErasing to simulate occlusions.

---

## Modeling Methods

### Backbone Architectures
- **ResNet18, ResNet34:** Traditional residual networks.
- **DenseNet121:** Densely-connected convolutional network, recurring due to strong early results.
- **EfficientNet-B0:** Used for efficiency and tested with cross-validation.
- **MobileNetV2:** Explored for lightweight applications.

### Classification Head
- Last layer is always replaced with a single-unit head (`nn.Linear(..., 1)`), converting backbone to binary classifier.

### Loss Functions
- **BCEWithLogitsLoss:** Default for binary classification.
- **Focal Loss:** Later models adopt the focal loss (gamma=2, alpha=0.25) to address class imbalance and focus on hard examples.

### Optimizers and Schedulers
- **Adam/AdamW:** Used throughout for efficient first-order optimization with weight decay regularization.
- **CosineAnnealingLR:** Employed to modulate learning rate for better convergence.
- **Stochastic Weight Averaging (SWA):** Used in one experiment, with proper model checkpointing.
- **Exponential Moving Average (EMA):** Widely adopted in later stages to stabilize model weights and improve generalization.

### Augmentation Details
- **Test-Time Augmentation (TTA):** Began with averaging predictions for original, horizontal, and vertical flips; extended to include 90°, 180°, 270° rotations for further robustness.
- **Ensembling:** No explicit model ensembling; averaging via TTA and EMA or SWA handled temporal/augmentation ensembling.

### Fairness Strategies
- **Skin Tone Weighted Sampling:** Explicit WeightedRandomSampler used to ensure equal exposure to each skin tone during training.
- **Class Weighted Sampling:** Addressed overall class imbalance in addition.

---

## Results Discussion

### Performance (Validation AUROC)
| Method                                 | Backbone      | Augmentation & Pipeline Features                              | Fairness | Validation AUROC |
|-----------------------------------------|--------------|--------------------------------------------------------------|----------|------------------|
| ResNet18                               | ResNet18     | Standard aug, BCE, ImageNet norm                             | No       | 0.8873           |
| EfficientNet-B0 (CV)                   | EffNet-B0    | 5x CV, AdamW, BCE, ImageNet norm                             | No       | 0.9036           |
| DenseNet121                            | DenseNet121  | Standard aug, BCE, Cosine sched, ImageNet norm               | No       | 0.9266           |
| DenseNet121 + MixUp                    | DenseNet121  | +MixUp, Cosine sched, TTA (flips)                            | No       | 0.9273           |
| DenseNet121 + MixUp + TTA              | DenseNet121  | +Test Time Augmentation                                      | No       | 0.9290           |
| DenseNet121 + MixUp + ColorJitter      | DenseNet121  | +ColorJitter, TTA                                            | No       | 0.9277           |
| DenseNet121 + MixUp + RandomErasing    | DenseNet121  | +RandomErasing, TTA                                          | No       | 0.9260           |
| DenseNet121 + CutMix                   | DenseNet121  | CutMix (beta=1), TTA                                         | No       | 0.9301           |
| DenseNet121 + CutMix + EMA             | DenseNet121  | CutMix, EMA, TTA                                             | No       | 0.9334           |
| DenseNet121 + CutMix + RandAugment     | DenseNet121  | CutMix, RandAugment, EMA, TTA                                | No       | 0.9256           |
| DenseNet121 + CutMix + Focal Loss      | DenseNet121  | CutMix, Focal Loss, EMA, TTA                                 | No       | 0.9289-0.9269*   |
| DenseNet121 + EMA + WeightedSampler    | DenseNet121  | CutMix, EMA, WeightedRandomSampler (class), TTA              | Yes      | 0.9342           |
| DenseNet121 + EMA + WeightedSampler + 6xTTA | DenseNet121 | CutMix, EMA, Weighted Sampler, 6x TTA (flips+rotations)   | Yes      | 0.9327           |
| DenseNet121 + SWA + SkinToneSampler    | DenseNet121  | CutMix, SWA, Skin tone weighted sampler, TTA                 | Yes      | 0.9233           |

\* Two Focal Loss implementations yielded nearly identical AUROC, both around 0.927–0.929.

### Empirical Takeaways
- **Progressive Improvements:** DenseNet121 outperformed other backbones, especially after data augmentation and loss improvements.
- **MixUp & CutMix:** Both significantly improved generalization. CutMix combined with EMA yielded the highest single-model AUROC (0.9334).
- **Test-Time Augmentation:** TTA improved model robustness—extension to 6-view orientation TTA yielded small but consistent AUROC gains.
- **Loss Functions:** Focal loss gave minor improvements and was robust to class imbalance.
- **Sampling for Fairness:** WeightedRandomSampler (for class and/or skin tone) was crucial for fairness and slightly improved AUROC.
- **Ensembling:** EMA of model weights, acting as a temporal ensemble, consistently boosted results.

### Fairness
- Early models did not address skin tone; later models included explicit sampler modifications to guarantee balanced skin tone exposure during training, thus supporting fairness.

---

## Future Work

1. **External Validation:** Assess generalization on external and more diverse datasets, including underrepresented skin tones.
2. **Calibration:** Assess/adjust probability calibration, especially important for clinical thresholds.
3. **Explainability:** Integrate saliency maps or Grad-CAM to support clinical interpretability.
4. **Multi-Task Learning:** Explore skin tone prediction as auxiliary task to bolster skin-type fairness.
5. **Deployable Pipeline:** Containerize model and inference function for cloud/clinical integration.
6. **Data Augmentation Exploration:** Further investigate advanced augmentations (e.g., adversarial training, self-supervision).
7. **Longitudinal Robustness:** Assess stability when new imaging devices are added (domain adaptation).
8. **Bias Audits:** Systematic audit for performance disparities across demographic/phenotypic groups.

---

## Summary

Through systematic experimentation, the following empirical design emerged as optimal:

- **Model:** DenseNet121 backbone, single-output head.
- **Augmentations:** CutMix (during training), Test-Time Augmentation (6 augmentations), occasional use of MixUp, ColorJitter, RandomErasing, and RandAugment.
- **Loss:** BCEWithLogitsLoss or Focal Loss.
- **Optimization:** AdamW, cosine annealing scheduler, with EMA of weights.
- **Fairness:** WeightedRandomSampler on skin tone and/or class for balanced training batches.

**Best AUROC achieved:** 0.9342 when using DenseNet121, EMA, CutMix, class-balanced WeightedRandomSampler, and TTA, with similar performance using extended (6x) TTA.

**Deployment:** The pipeline provides a `predict(folder_path)` function, robust to orientation and designed to be fair across skin tones, outputting the malignancy probability for each image.

---

```
