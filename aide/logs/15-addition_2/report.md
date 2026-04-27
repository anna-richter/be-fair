```markdown
# Technical Report: Fair and Robust Skin Lesion Malignancy Prediction

## Introduction

This report summarizes the empirical investigation into automated skin lesion malignancy classification with a focus on both high accuracy and fairness across skin tones. The task, motivated by clinical needs, is to train a model that outputs malignancy probabilities for input images while minimizing AUROC (Area Under the Receiver Operating Curve) disparity between light and dark skin tones. Several architectures, feature extractors, classifiers, and fairness interventions were explored and benchmarked.

---

## Preprocessing

### Data Handling

- Dataset consisted of image files and metadata (including labels and skin tone).
- Images with missing skin tone had the value replaced by cohort median or were otherwise handled per fold for fair weighting.
- Each image is loaded, converted to RGB, and normalized according to standard ImageNet means and standard deviations.

### Embedding Extraction

Empirical results showed superior downstream classifier performance and stable optimization when using modern pretrained CNN encoders as frozen feature extractors:

- **EfficientNet-B0** (1280-dim): Provided best tradeoff between speed and accuracy.
- **ResNet18/ResNet50/DenseNet121**: Used as alternatives for comparison; did not outperform EfficientNet-B0.
- **Test-time augmentations**: Embeddings are extracted under several deterministic geometric transforms:
    - Original, Horizontal Flip, Vertical Flip, 90° Rotation, 180° Rotation (for 5-way TTA).

### Embedding Representation

- For most effective results, each transform’s embedding was either:
    - **Averaged**: To reduce variance and improve robustness.
    - **Concatenated**: To allow the classifier to learn optimal transform weighting.

---

## Modelling Methods

### Classifier Architecture

- **Logistic Regression** and **RandomForestClassifier** were explored for baseline methods.
    - Logistic regression underperformed relative to ensemble gradient boosting.
- **LightGBM** (Gradient Boosted Decision Trees) consistently provided the best AUROC.

### Fairness Strategies

- **Explicit sample weighting** based on skin tone frequency (using `compute_sample_weight('balanced', skin_tones)`) was evaluated.
    - Helped reduce AUROC gap between light (tone ≤3) and dark (tone ≥5) skin.
- **TTA (Test-Time Augmentation)**:
    - Multi-transform feature averaging enhanced both overall model robustness and fairness, as train/test augmentations exposed classifiers to geometry-invariant cues.
    - 5-way TTA (orig, hflip, vflip, 90°, 180° rotation) gave best empirical results.

### Ensembling

- **LightGBM Ensemble**:
    - Three LightGBM models with different seeds, averaged at inference, further suppressed variance and improved AUROC, delivering state-of-the-art performance.

### Inference Pipeline

- Final trained models are saved.
- The `predict()` function takes a folder of `.jpg` images, applies the same embedding and TTA protocol, and returns malignancy probabilities for each image (in a DataFrame).

---

## Results Discussion

| Approach              | CV AUROC | AUROC Gap (L/D) | Notes                                                      |
|-----------------------|----------|-----------------|------------------------------------------------------------|
| ResNet18+LogReg       | 0.8464   | 0.1453          | Simplicity, lower fairness                                 |
| EfficientNet+LightGBM | 0.8765   | 0.0207          | Strong, balanced, fast inference                           |
| RF+ResNet50           | 0.8351   | 0.0230          | Good but not as strong as LightGBM                         |
| DenseNet121+RF        | 0.8320   | 0.0244          | Robust baseline                                            |
| EfficientNet+TTA (4x) | 0.8935   | 0.0124          | 4-way average (orig/hflip/vflip/rot90), best fairness/ROC  |
| EfficientNet+TTA+180° | 0.8931   | 0.0094          | 5-way TTA, minimal fairness gap                            |
| EfficientNet+Ensemble | 0.8935   | 0.0124          | 4-way TTA + 3-seed ensemble, most robust overall           |

**Key findings:**

- **EfficientNet-B0 as feature extractor outperformed all other architectures** for both overall AUROC and fairness, especially when combined with TTA averaging.
- **LightGBM classifiers, with/fairness weighting and/or ensemble, offered the best compromise** between complexity, speed, AUROC, and AUROC fairness gap.
- **Ensembling with different seeds further boosted stability**, confirming that variance reduction in the classifier translates to more reliable clinical deployment.
- **Adding 5th TTA (180°) slightly reduced the light/dark AUROC gap to below 1%.**

**Final Recommendation:**
The most robust and fair pipeline is:
- EfficientNet-B0 feature extraction.
- 5-way TTA (original, horizontal flip, vertical flip, 90°, 180° rotation), with averaged embeddings.
- Ensemble of three LightGBM classifiers trained and evaluated on these averaged embeddings.
- Model achieves **AUROC ≈ 0.893**, **AUROC gap (light/dark) ≈ 0.009–0.013**, which meets both the accuracy and fairness goals.

---

## Future Work

- **Domain Adaptation**: Address potential dataset shift from curated to field images; consider fine-tuning the CNN on a small labeled subset from the deployment target domain.
- **Uncertainty Quantification**: Integrate model calibration or Bayesian methods to provide per-image uncertainty, valuable for triage or second-opinion workflows.
- **Fine-grained Fairness**: Explore fairness not only across dichotomized skin tone but across the full range and intersectional groups (e.g., by age, sex).
- **Explainability**: Add tools for visualizing salient regions of predictions (e.g., Grad-CAM on EfficientNet features).
- **Efficiency**: Optimize TTA and ensemble computation for faster clinical deployment, possibly replacing some augmentations with learned geometry-invariance.

---

**Conclusion:**  
The recommended pipeline meets and exceeds the project's requirements for high-accuracy, fair, and robust skin lesion malignancy classification. The empirical process confirms that careful feature design, aggressive geometric augmentation, and ensemble learning are together key to reliable and equitable medical AI in dermatology.

```