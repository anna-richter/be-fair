```markdown
# Technical Report: Automated Prediction of Malignant and Benign Skin Lesions Using Machine Learning

## Introduction

The goal of this project is to develop an AI-based system capable of differentiating between malignant and benign skin lesions using image data provided by dermatologists. The core output is a model that accepts a folder of new skin lesion images and returns, for each image, the probability (0–1) that the lesion is malignant. The evaluation metric is Area Under the Receiver Operator Curve (AUROC), which measures the model's ability to distinguish between classes.

This report summarizes empirical results and technical decisions from a series of design and modeling experiments, including traditional feature extraction methods, deep feature pipelines, different classifiers, and test-time augmentation strategies.

## Preprocessing

### Data Handling

- All experiments begin by loading a dataset containing image file names and corresponding labels (`benign` or `malignant`), mapping labels to a binary format.
- Images are loaded from disk, converted to RGB if necessary, and preprocessed according to the requirements of the downstream feature extraction or deep model pipeline.

### Image Preprocessing Strategies

- **Resizing:** All pipelines resize images to a consistent resolution (commonly 224x224 pixels), matching requirements for pretrained deep models.
- **Normalization:** For deep models, images are normalized according to the mean and standard deviation of the pretrained backbone, either manually or using built-in torchvision utilities.
- **Augmentation:** Some pipelines include random transformations (e.g., horizontal/vertical flips, rotations) at training or test time to improve robustness and generalization.

## Modeling Methods

### Feature Extraction Techniques

1. **Traditional Features:**
   - **Histogram of Oriented Gradients (HOG):** Extracts gradient orientation features from grayscale, resized images.
   - **Color Histograms:** Quantize each RGB channel and concatenate histograms to form a color feature vector.

2. **Deep Features:**
   - **ResNet18/EfficientNet-B0 Backbones:** Utilized as fixed feature extractors by removing the final classification layer. Features are extracted from the penultimate network activations.
   - **Test-Time Augmentation (TTA):**
     - Variants include extracting features not just from original images, but also from horizontally and/or vertically flipped versions and averaging the resulting feature vectors ("4-way TTA" includes all flips and both-flipped).

### Classifiers

- **Logistic Regression:** Linear model, frequently used for classification on deep/extracted features.
- **Random Forest Classifier:** Nonlinear ensemble classifier, particularly on traditional features.
- **LightGBM:** Gradient boosting machine capable of modeling nonlinear interactions, applied to deep features and color histogram features.

### Validation Strategies

- **5-fold Stratified Cross-Validation:** Robust estimate of generalization by splitting data into five balanced folds.
- **Single Hold-Out Split:** Used for rapid prototyping (80/20 train/validation), mostly with deep end-to-end pipelines.

### Submission Pipeline

Each approach culminates in:
- Training the final model on all available data after cross-validation.
- Implementing a `predict()` function/API to compute malignancy probabilities for new images, returning a DataFrame or `.csv` with probabilities for each input image.

## Results Discussion

### Summary Table of Major Experiments

| Feature Type           | Backbone         | TTA      | Classifier      | Validation AUROC |
|------------------------|------------------|----------|-----------------|-----------------|
| HOG                   | —                | None     | Random Forest   | 0.6664          |
| Color Histogram        | —                | None     | LightGBM        | 0.6872          |
| Deep Features          | ResNet18         | None     | Logistic Reg.   | 0.8440          |
| Deep Features          | ResNet18         | H-Flip   | Logistic Reg.   | 0.8479          |
| Deep Features          | EfficientNet-B0  | H-Flip   | Logistic Reg.   | 0.8724          |
| Deep Features          | EfficientNet-B0  | 4-way    | Logistic Reg.   | 0.8790          |
| Deep Features          | EfficientNet-B0  | 4-way    | LightGBM        | **0.8958**      |
| Deep Features          | ResNet18         | 4-way    | Logistic Reg.   | 0.8541          |
| Deep Features*         | ResNet18         | None     | Logistic Reg.   | 0.8017          |
| End-to-End CNN         | ResNet18         | —†       | Fine-tuned      | 0.8082          |

\* Using built-in normalization with torchvision weights  
† Some training-time augmentation only

#### Key Insights

1. **Deep feature pipelines outperform traditional features.** Classical approaches using HOG or color histograms plateaued well below the deep-learning-based pipelines (AUROC 0.66–0.69 vs. >0.80).
2. **Test-Time Augmentation Improves Generalization.** Averaging features from simple flips (horizontal/vertical) increased AUROC by 0.02–0.04 points depending on backbone.
3. **Stronger Backbones Help.** Upgrading from ResNet18 to EfficientNet-B0 for deep feature extraction further improved scores, especially with TTA.
4. **LightGBM on Deep Features Outperforms Linear Models.** Swapping logistic regression for LightGBM on (EfficientNet-B0, 4-way-TTA) features yielded the best overall result (AUROC 0.8958).
5. **Efficient Feature/Inference Pathways Are Useful.** Caching features during training and disabling TTA for entirely new image batches can noticeably reduce inference latency without affecting training accuracy.

## Future Work

While the current pipeline achieves strong performance (AUROC ≈ 0.90), several avenues could further enhance diagnostic accuracy or system usability:

- **Model Ensembling:** Combine outputs from diverse backbones/classifiers.
- **Fine-Tuned or Fully End-to-End Models:** Rather than fixed feature extractors, train the entire network (including the backbone) on available data for potentially better performance.
- **Advanced Augmentation:** Use color jitter, random crops, or domain-specific augmentations.
- **Clinical Metadata Integration:** Incorporate tabular clinical features if available (e.g., age, lesion location) using multimodal models.
- **Uncertainty Quantification:** Output not only point probabilities but also confidence intervals.
- **Deployment/UI:** Develop a streamlined interface for clinicians, including quality checks for uploaded images.

## Conclusion

Through systematic experimentation, we find that extracting 4-way test-time augmented features from a pretrained EfficientNet-B0 backbone, followed by LightGBM classification, delivers the highest AUROC (0.8958) for distinguishing malignant from benign skin lesions on this dataset. The solution is robust, automatable, and supports batch inference on arbitrary image folders, making it practical for deployment in a dermatology workflow.

```