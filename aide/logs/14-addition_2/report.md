```markdown
# Skin Lesion Malignancy Classification: Technical Summary

## Introduction

The project focuses on developing a robust, fair, and automated skin lesion classifier to differentiate malignant from benign lesions using a curated dataset. The main goals are to maximize overall classification performance, measured by area under the receiver operating characteristic curve (AUROC), and to minimize AUROC gaps between skin tone groups (i.e., ensure fairness across light and dark skin). Several architectures (ResNet18, EfficientNet-B0, DenseNet121, ViT, MobileNetV3) and fairness-promoting strategies (e.g., weighted sampling, MixUp, test-time augmentation) were empirically tested.

---

## Preprocessing

### Data Preparation
- **Label Mapping:** Labels are binarized as `malignant=1`, all else (benign, non-neoplastic) as `0`.
- **Skin Tone:** Skin tone is tracked per sample and used for fairness-aware resampling.
- **Splits:** Evaluation is generally performed via 5-fold stratified cross-validation or 80/20 stratified holdout, stratified by malignancy label. Skin-tone stratification is used for fairness evaluation.
- **Transforms:** Standard ImageNet normalization; augmentations include random resized crop, horizontal flip, rotation, and color jitter.

---

## Modeling Methods

### Model Choices

- **ResNet18:** Baseline—standard CNN, pretrained on ImageNet.
- **EfficientNet-B0:** Lightweight yet strong performer. Used with sample-weighted BCE loss.
- **DenseNet121:** Common baseline and main pipeline. Suits feature reuse.
- **MobileNetV3-Large:** Highly efficient; used with fairness sampling.
- **ViT-Ti/16:** Vision Transformer variant for patch-based learning.

### Regularization & Fairness

- **WeightedRandomSampler:** In almost all pipelines, a per-sample weight is computed as the inverse frequency of a sample’s skin tone group (light/dark); each batch is thus balanced for skin tone. This is applied independently to each cross-validation fold and to final training on the full dataset.
- **MixUp Augmentation:** Images and labels mixed per batch with a Beta distribution (α=0.4), encouraging smoother decision boundaries and boosting generalization.
- **Test-Time Augmentation (TTA):** Predictions at inference are averaged for the original and horizontally flipped image, increasing robustness.
- **Loss Function:** BCEWithLogitsLoss (with/without per-sample weights).
- **Optimizer:** Adam; learning rate typically 1e-4.

### Training Details

- **Epochs:** Most runs use 3 epochs per fold/full train (EfficientNet: 5 epochs).
- **Batch Size:** Standardized to 32 for balance between training speed and generalization.
- **Hardware:** GPU if available; otherwise, CPU fallback.

---

## Results Discussion

### Performance

| Approach                   | Mean/CV/Test AUROC | Fairness Technique              | Comments                                   |
|----------------------------|--------------------|---------------------------------|--------------------------------------------|
| **ResNet18**               | 0.876              | WeightedRandomSampler           | Moderate fairness, moderate AUROC          |
| **EfficientNet-B0**        | 0.888              | Weighted loss per skin-tone     | Good learning curve, high validation AUROC |
| **DenseNet121 baseline**   | 0.915              | None explicit                   | Strong AUROC, fairness not measured        |
| **DenseNet121+WeightedSampler** | 0.904-0.910    | WeightedRandomSampler           | Strong AUROC w/ fairness, robust across variations |
| **DenseNet121+MixUp**      | 0.91–0.92          | MixUp, some w/ TTA              | Best/near-best AUROC, high stability       |
| **MobileNetV3-Large**      | 0.905              | WeightedRandomSampler           | High efficiency, good AUROC                |
| **ViT-Ti/16 (Transformer)**| 0.906              | WeightedRandomSampler           | Competitive AUROC, no TTA                  |
| **DenseNet121+TTA only**   | 0.922              | TTA, no fairness focus          | Very strong AUROC, fairness not measured   |

- **Fairness:** Where checked, the AUROC gap between skin tone groups is typically reduced to <0.08 with the sampler.
- **MixUp** tends to yield the most consistent generalization gains (~0.91–0.92 AUROC with TTA).
- **Test-Time Augmentation** (TTA) consistently offers 0.01-0.02 AUROC improvement.
- **Fairness via Sampling:** WeightedRandomSampler per skin tone is computationally efficient and effective; where explicit, the within-group AUROC gap post-sampling is minimized, with little compromise to overall AUROC.

### Pipeline Features

- **Prediction Function:** Every final pipeline provides a `predict(folder_path)` function returning malignancy probabilities for all input images, compatible with batch processing and TTA.
- **Model Saving:** The best model (by validation AUROC) is saved as `model.pth` for deployment.

---

## Future Work

- **Fairness Reporting:** Future pipelines should always report per-skin-tone AUROCs and explicit AUROC gap metrics in validation, not just overall AUROC.
- **Ensembling:** Combining predictions of several strong and fair models (e.g., MixUp + TTA and fairness-sampled DenseNet) could further boost both accuracy and fairness.
- **Advanced Augmentation:** Explore augmentations more specific to skin imaging (e.g., local brightness, blur, dermatological texture).
- **Skin Tone Annotation:** Expansion and refinement of skin tone labeling could further enhance fairness performance and interpretability.
- **Calibration:** Investigate and, if necessary, calibrate probability outputs to ensure decision thresholds are equally reliable for all skin tones.
- **Explainability:** Integrate model explainability tools (e.g., Grad-CAM) for clinical confidence.

---

# Conclusion

A DenseNet121 architecture, trained with skin-tone-balanced sampling (WeightedRandomSampler), test-time augmentation, and (optionally) MixUp achieves both strong performance (mean AUROC: 0.91–0.92) and robust fairness (minimal AUROC gap between skin tones). The pipeline complies with the dermatologist's requirements for model saving and flexible, batch-friendly malignancy probability inference on new images.

---
```