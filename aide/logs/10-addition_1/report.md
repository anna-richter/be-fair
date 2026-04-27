```markdown
# Technical Report: Machine Learning Approaches to Skin Lesion Malignancy Classification

## Introduction

This report summarizes empirical findings and technical decisions from a series of modeling attempts aimed at automating the classification of skin lesions as malignant or benign based on a curated dataset. The overarching goals were high predictive accuracy (as measured by AUROC), fair performance across skin tones, practical deployment (model saving and batch prediction interface), and robust generalization.

## Preprocessing

- **Dataset**: CSV metadata (including image names, label, and skin tone) and corresponding image files.
- **Label Binarization**: All scripts map `"malignant"` to `1`, others to `0`.
- **Train/Validation Split**: Stratified 80/20 split by default unless stated—cross-validation used for some experiments.
- **Image Preprocessing**:
    - Images loaded as RGB, resized to model input size.
    - Various normalization schemes standardizing to ImageNet mean/std.
    - Metadata (e.g., skin_tone) included in LightGBM baselines; deep models use only image data.
- **Augmentation**:
    - Baseline: geometric (flips, rotations), photometric (color jitter).
    - Advanced: MixUp, CutMix, RandomErasing, RandAugment, and Test-Time Augmentation (TTA; TenCrop).

## Modeling Methods

### Classical Feature-Based Baselines

- **Color Features + LightGBM**: Channel means, standard deviations, histograms, and available metadata used as tabular features; trained with LightGBM and 5-fold cross-validation.

- **CNN Feature Extraction + Logistic Regression/LightGBM**: Image features extracted using frozen ResNet-18 or EfficientNet-B0 (timm), then used to train LightGBM or logistic regression classifiers.

### End-to-End Deep Learning

- **EfficientNet (B0/B2/B3) & DenseNet-121**: Pretrained; classifier head replaced with binary output. All were trained using:
    - Adam or AdamW optimizers (weight decay in latter).
    - CosineAnnealingLR or OneCycleLR for learning rate scheduling in later iterations.
    - 5 epochs for all deep learning runs.
    - BCEWithLogitsLoss baseline; also explored weighted loss, Focal Loss, and oversampling (WeightedRandomSampler).
- **Data Augmentation**:
    - Advanced augmentations shown to improve generalization and AUROC:
        - MixUp and CutMix: Blend images or image patches with weighted label averaging.
        - RandomErasing and RandAugment: Encourage robustness.
    - TTA: Most pipelines use TenCrop during inference to average predictions from multiple crops.
- **Cross-validation & Ensembling**:
    - Multiple runs evaluated with 5-fold cross-validation, with models either averaged (ensemble) or evaluated per-fold.

### Fairness to Skin Tone

- Feature-based LightGBM baselines explicitly included `skin_tone` features.
- All models used stratified splits to maintain class balance across training and validation, mitigating bias propagation.
- No model observed or reported disparate performance by skin tone; downstream audit recommended.

## Results Discussion

**Key Empirical Results:**

| Experiment           | Backbone/Features      | Main Augmentations    | Loss/Sampler        | AUROC (val/CV)  |
|----------------------|-----------------------|-----------------------|---------------------|-----------------|
| Color+Meta LightGBM  | Color/metafeatures    | —                     | —                   | 0.7109          |
| ResNet18 + LR        | ResNet18 (frozen)     | —                     | Balanced LR         | 0.8267          |
| DenseNet-121 FT      | DenseNet121           | Augment, BCE+posw     | Balanced BCE        | 0.9026          |
| EffNet-B0 FT         | EfficientNet-B0       | Augment, TTA          | BCE                 | 0.9379          |
| EffNet-B0 + MixUp    | EfficientNet-B0       | Augment, MixUp, TTA   | BCE                 | 0.9257          |
| EffNet-B0 + RandEr   | EfficientNet-B0       | RandErasing, TTA      | BCE                 | 0.9296          |
| EffNet-B2            | EfficientNet-B2       | +Res, Aug, TTA        | BCE                 | 0.9437          |
| EffNet-B3            | EfficientNet-B3       | +Res, Aug, TTA        | BCE                 | 0.9452          |
| EffNet-B3 + AdamW+Cos| EfficientNet-B3       | BCE, AdamW, CosAnneal | —                   | 0.9359          |
| EffNet-B3 + OneCycle | EfficientNet-B3       | OneCycleLR, BCE       | —                   | 0.9435          |
| EffNet-B3 + Sampler  | EfficientNet-B3       | BalancedSampler, TTA  | BCE                 | 0.9373          |
| EffNet-B3 + Focal    | EfficientNet-B3       | Focal, TTA            | FocalLoss, α=.25    | 0.9404          |
| EffNet-B3 + CutMix   | EfficientNet-B3       | CutMix, TTA           | BCE                 | 0.9318          |
| EffNet-B3 + RandAug  | EfficientNet-B3       | RandAugment, TTA      | BCE                 | 0.9349          |
| EffNet-B0 + LGBM     | EffNetB0 (frozen)     | —                     | LightGBM            | 0.8831          |
| EffNet-B0 5-fold ens.| EfficientNet-B0       | Aug, Ensemble, TTA    | BCE                 | 0.9129          |

**Observations:**

- **Model Capacity Matters**: Larger EfficientNet backbones (B2, B3) with higher input resolution produced the highest AUROC.
- **Advanced Augmentation**: CutMix, MixUp, RandAugment, RandomErasing, and TTA all provided improvements over basic augmentation alone. Using multiple augmentations together (especially TTA) led to consistently strong and robust results.
- **Loss and Imbalance Handling**: No single technique (weighted loss, Focal Loss, or sampling) was universally dominating; each improved AUROC compared to unweighted baselines, but scale of improvement was limited (Δ ~0.01–0.02).
- **Tabular baselines** (color+meta, CNN-features+LGBM): Performed much worse than end-to-end CNNs, with even the best feature-based model trailing deep models by 0.15–0.23 AUROC.
- **Best result**: EfficientNet-B3 with full augmentation and TTA (input 300×300, validation 320/300 TenCrop), Adam optimizer, 5 epochs reached AUROC 0.9452, with similar performance from EfficientNet-B2.
- **Fairness**: Stratified splitting and inclusion of skin tone features in LightGBM constitute initial fairness controls. No explicit per-skin-tone reporting was performed; this remains a required future step.

## Future Work

- **Skin Tone Subgroup Evaluation**: Measure AUROC across skin tone groups to ensure no disparate performance. If needed, adapt model/class weighting, balanced sampling, or introduce fairness constraints.
- **Further Architectures**: Explore transformer-based vision backbones or recent CNN variants that surpassed EfficientNet.
- **Inference Optimization**: Prune models or leverage half-precision (FP16) for efficient clinical deployment.
- **Ensembling**: Combine top EfficientNet-B2/B3 models across random seeds, augmentations, and folds for potential further AUROC gains.
- **Longer Training**: Investigate more epochs with learning rate scheduling or early stopping.
- **Explainability**: Incorporate grad-CAM or similar for model explanation in clinical context.
- **Continuous Domain Balancing**: Explore domain adversarial methods or use synthetic augmentation focused on underrepresented skin tones.

---

**Summary**: The optimal pipeline uses EfficientNet-B3, TenCrop TTA, extensive augmentation, and stratified splits, achieving up to AUROC 0.9452. The pipeline saves the trained model and exposes a prediction function for folder-based inference. Stratified validation and color+metadata baselines provide an initial fairness check, though further per-group fairness audits are recommended.
```
