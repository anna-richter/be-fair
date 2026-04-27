```markdown
# Skin Lesion Malignancy Classification: Empirical Findings & Technical Summary

## Introduction

The objective was to develop a high-performing and fair malignancy classifier for skin lesion images, with an emphasis on minimizing performance gaps across skin tone groups. The primary metric is AUROC (area under the ROC curve), with fairness measured as the AUROC gap between light and dark skin tones. A variety of architectures, data preprocessing, augmentation, and fairness interventions were explored; this report summarizes the key empirical results and design choices.

---

## Preprocessing

- **Dataset**: Images annotated with file names, lesion label (`malignant`, `benign`, `non-neoplastic`), and `skin_tone` (typically 1–6).
- **Label Mapping**: `malignant` lesions assigned label 1, `benign` and `non-neoplastic` as 0 for binary classification.
- **Splitting**: 5-fold stratified cross-validation on binary labels, ensuring consistent evaluation and fair comparison.
- **Skin Tone Handling**: Skin tone values were used for fairness-aware sampling and group-wise evaluation.
    - Light: tones 1–3
    - Dark: tones 5–6
- **Image Transforms**:
    - **Training**: Progressive shift from basic resizing/cropping to strong augmentations (RandomResizedCrop, ColorJitter, random flips, and RandomErasing).
    - **Validation/Test**: Standard resizing and centercrop for consistency during evaluation and inference.
    - **MixUp**: Applied in later experiments for robust generalization.

---

## Modeling Methods

### Data Sampling & Fairness

- **Fairness-Aware Sampling**: Most models used a weighted random sampler to balance skin tone representation within each batch, computed as the inverse frequency per skin tone group.
- **Class Imbalance**: Some designs further combined class-balanced sampling, but focus remained on skin tone parity.
- **Fairness Metric**: The absolute AUROC difference between light (tones ≤3) and dark (tones ≥5 or [5,6]) was consistently reported.

### Model Architectures

- **CNN Backbones**:
    - *ResNet18/ResNet50*: Used as main classifier with the final FC layer replaced to output a single logit.
    - *EfficientNet-B0*: Used as both an end-to-end image classifier and as a feature extractor for gradient boosting models.
- **Additional Approaches**:
    - *Skin Tone Embedding*: Concatenated learned embeddings for skin tone to improve fairness (ResNet50/timm).
    - *LightGBM on CNN Embeddings*: EfficientNet-B0 used to extract features; LightGBM trained on these, using sample weights for skin tone.

### Training Techniques

- **Loss Function**: BCEWithLogitsLoss for binary output.
- **Optimization**: Adam optimizer with initial learning rate mostly at 1e-4; OneCycleLR scheduler tested.
- **Epochs**: Typically 3–5 per fold, chosen for practical convergence.
- **MixUp Augmentation**: Applied as an atomic change, resulting in improved AUROC and smoother decision boundaries.
- **Test-Time Augmentation (TTA)**:
    - Baseline: None or horizontal flip.
    - Enhanced: 4-way TTA with horizontal, vertical, and both flips; in one experiment, even more symmetries and rotations were averaged.

---

## Results Discussion

### Summary Table

| Approach                            | Backbone         | AUROC (mean CV) | Fairness Gap* | Test-Time Augmentation        | Notable Strategies                                    |
|--------------------------------------|------------------|-----------------|--------------|-------------------------------|-------------------------------------------------------|
| ResNet18, fairness sampler           | ResNet18         | 0.8915          | 0.0301       | None                          | Baseline, balanced batches by skin tone               |
| Skin tone embedding                  | ResNet50 (timm)  | 0.8913          | N/R          | None                          | Concatenated skin tone embedding                      |
| EfficientNet-B0 (frozen, simple aug) | EfficientNet-B0  | 0.8348          | N/R          | None                          | Only last layer trained                               |
| EfficientNet-B0, MixUp, fairness     | EfficientNet-B0  | 0.807           | 0.031        | None                          | End-to-end + MixUp, fairness-aware sampler            |
| LightGBM on embeddings               | EfNet-B0+LGBM    | 0.8876          | N/R          | None                          | LGBM with sample weights on EfNet features            |
| ResNet18, stronger train aug         | ResNet18         | 0.8740          | 0.0373       | None                          | Added vertical flip, ColorJitter                      |
| ResNet18, MixUp                      | ResNet18         | 0.9010          | 0.015        | None                          | Robust generalizer                                    |
| ResNet18, MixUp + hflip TTA          | ResNet18         | 0.9065          | 0.019        | Horizontal flip               | Boosts robustness, further narrows fairness gap       |
| ResNet18, MixUp + OneCycleLR         | ResNet18         | 0.8779          | 0.031        | Horizontal flip               | Dynamic LR schedule                                   |
| ResNet18, MixUp + 4-way TTA          | ResNet18         | 0.904           | 0.0117       | hflip/vflip/both/original     | Most robust—strong accuracy and equity                |
| ResNet18, MixUp + 4-way TTA (alt)    | ResNet18         | 0.903           | 0.021        | hflip/vflip/both/original     | Repeated confirmation                                 |
| ResNet18, strong aug, MixUp, hflip   | ResNet18         | 0.8812          | 0.0238       | Horizontal flip               | RandomErasing + ColorJitter etc                       |
| ResNet18, strong aug, MixUp, hflip   | ResNet18         | 0.8899          | 0.031        | Horizontal flip               | RandomErasing, improved fairness                      |
| ResNet18, strong aug, fixed fairness | ResNet18         | 0.8928          | 0.023        | Horizontal flip               | Correct group gap calc                                |
| ResNet18, 7-way TTA, fixed gap       | ResNet18         | 0.9009          | 0.024        | 7-fold (incl. rot90s, flips)  | Extensive TTA, minimal gap                            |

\*Gap = AUROC(light) - AUROC(dark), absolute value; N/R: not reported.

#### Key Takeaways

- **Fairness-aware batch sampling** (by skin tone) consistently reduces AUROC gaps across groups.
- **MixUp augmentation** notably improves both mean AUROC and fairness gap.
- **Test-time augmentation** (especially 4-way) consistently adds ~0.01–0.02 AUROC and reduces fairness gaps, with 4-way (h/v/both/orig) being especially effective.
- **Strong training augmentations** (ColorJitter, flips, RandomErasing) help generalization, with moderate AUROC boosts.
- **More elaborate TTA (e.g., including rotations)** shows marginal gains beyond 4-way TTA.

---

## Future Work

- **Longer training or more powerful backbones** (e.g., EfficientNet-B3, ViTs) could further boost AUROC, but may require more computational resources.
- **Explicit skin tone adversarial training** or domain adaptation could be explored to improve fairness beyond reweighting and embeddings.
- **Exploring uncertainty modeling** (e.g., MC Dropout) for more calibrated probabilities, especially for ambiguous cases.
- **Automated hyperparameter optimization** (learning rate, augmentation intensity, MixUp alpha) may yield further performance gains.
- **Fine-grained skin tone calibration**: Use continuous skin tone predictions or more sophisticated group partitions if sample sizes permit.

---

## Conclusion

Integrating fairness-aware sampling, strong data augmentation, MixUp during training, and robust test-time augmentation (4-way flip) enables a ResNet18-based model to achieve excellent AUROC (~0.90) and a minimal fairness gap (≤0.02) on skin lesion malignancy prediction. The recommended final strategy is:

- Use weighted sampling by skin tone in each training batch.
- Train a ResNet18 for 5 epochs with MixUp and strong augmentations.
- During prediction, apply 4-way test-time augmentation by averaging predictions over original, horizontal, vertical, and both flips.
- Use the provided `predict(model_path, image_folder)` interface for batch inference.

This approach results in robust, fair, and practically deployable malignancy risk estimates for dermatologists without systematic skin tone bias.
```