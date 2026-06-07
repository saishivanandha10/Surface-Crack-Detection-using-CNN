<div align="center">

# 🔍 Surface Crack Detection Using CNN

**Binary image classification for structural surface crack detection**  
Built with TensorFlow / Keras · Custom CNN · MobileNetV2 · ResNet50

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Kaggle](https://img.shields.io/badge/Run%20on-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com)

</div>

---

## Overview

This project trains and evaluates three deep learning models to detect surface cracks in concrete imagery — a critical task in structural health monitoring. The notebook covers the full ML pipeline: exploratory data analysis, preprocessing, augmentation, model training, Grad-CAM interpretability, error analysis, and deployment risk assessment.

| Model | Accuracy | F1-Score | AUC-ROC | Parameters | Train Time |
|-------|----------|----------|---------|------------|------------|
| **Custom CNN** | **99.75%** | **0.9975** | **0.9999** | 0.67 M | 32.8 min |
| MobileNetV2 | 31.97% | 0.4449 | 0.2349 | 2.60 M | 8.1 min |
| ResNet50 | 44.47% | 0.5376 | 0.3922 | 24.13 M | 6.5 min |

> **Key finding:** The lightweight custom CNN (0.67 M parameters) outperforms both transfer learning models by a large margin, achieving near-perfect 99.75% accuracy — demonstrating that a purpose-built architecture beats ImageNet-pretrained models on this specialized texture-based task.

---

## Research Questions

| ID | Question |
|----|----------|
| **RQ1** | How accurately can CNN-based models detect surface cracks on publicly available datasets? |
| **RQ2** | Which model family — custom CNN or transfer learning — gives the best performance-efficiency trade-off? |
| **RQ3** | How do preprocessing, augmentation, and class-balance handling influence model robustness? |
| **RQ4** | Do Grad-CAM explanations show models attending to meaningful crack-related regions? |
| **RQ5** | What are the main failure patterns, deployment limitations, and practical risks? |

---

## Repository Structure

```
Surface-Crack-Detection-using-CNN/
├── Surface_Crack_Detection_using_CNN.ipynb   ← Main notebook (10 sections)
├── Instructions.md                           ← Run instructions
├── Dataset link                              ← Kaggle dataset reference
├── Outputs/                                  ← Generated outputs
│   ├── figures/                              ← 20 PNG plots
│   ├── models/                               ← Saved .keras model files
│   └── reports/                              ← CSV evaluation reports
└── README.md
```

---

## Dataset

**[Surface Crack Detection — arunrk7 on Kaggle](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection)**

| Property | Value |
|----------|-------|
| Total images | 40,000 |
| Classes | `Positive` (cracked) · `Negative` (intact) |
| Class balance | Perfectly balanced — 20,000 per class |
| Image size | 227 × 227 px, JPEG |
| Split used | 70% train / 15% val / 15% test (stratified) |

---

## Models

### 1. Custom CNN
A 4-block convolutional network built from scratch:
- 4× `Conv → BatchNorm → MaxPool` blocks with progressive filter doubling (32 → 64 → 128 → 256)
- `GlobalAveragePooling2D → Dense(256, ReLU) → Dropout(0.5) → Sigmoid`
- Trained with early stopping, ReduceLROnPlateau, and best-checkpoint saving

### 2. MobileNetV2 (Transfer Learning)
- **Stage 1:** Feature extraction only — base frozen, 10 epochs
- **Stage 2:** Fine-tune top 30 layers — up to 20 additional epochs
- Custom head: `GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.3) → Sigmoid`

### 3. ResNet50 (Transfer Learning)
- **Stage 1:** Feature extraction only — base frozen, 10 epochs
- **Stage 2:** Fine-tune top 20 layers — up to 20 additional epochs
- Custom head: `GlobalAveragePooling2D → Dense(256, ReLU) → Dropout(0.4) → Sigmoid`

---

## Key Results

### RQ1 & RQ2 — Accuracy vs. Efficiency
The custom CNN achieves 99.75% accuracy at just 0.67 M parameters — the lowest count of the three models. Transfer learning models underperformed significantly, likely due to domain mismatch between ImageNet pretraining and low-texture crack surface imagery.

### RQ3 — Preprocessing & Augmentation
- Pixel normalization to `[0, 1]` (CNN) and model-specific ranges (MobileNetV2 / ResNet50) stabilized training
- Augmentation applied: random horizontal/vertical flips + brightness/contrast variation
- No class weighting required — the dataset is perfectly balanced
- Stratified splits preserved class ratios across all three sets

### RQ4 — Grad-CAM Interpretability
- Custom CNN heat maps correctly focus on crack texture regions
- MobileNetV2 shows finer-grained localization consistent with depthwise separable convolution
- Both models attend to crack-relevant features rather than background artefacts

### RQ5 — Failure Analysis & Deployment Risks

| Metric | Value |
|--------|-------|
| False Positives | 4 |
| False Negatives | 11 |
| High-confidence wrong predictions | 9 |
| Recommended classification threshold | < 0.5 (prioritize recall) |

**Deployment risks to monitor:** domain shift (outdoor/wet surfaces), lighting variation, image blur, and sensor noise not represented in the training set. Continuous monitoring and periodic retraining are strongly recommended.

---

## Outputs Generated

```
outputs/
├── figures/
│   ├── 01_sample_images.png
│   ├── 02_class_distribution.png
│   ├── 03_augmentation_examples.png
│   ├── 04_cnn_training_curves.png
│   ├── 05_cnn_confusion_matrix.png
│   ├── 06_cnn_roc_curve.png
│   ├── 07_cnn_prediction_grid.png
│   ├── 08_mnv2_training_curves.png
│   ├── 09_mnv2_confusion_matrix.png
│   ├── 10_mnv2_roc_curve.png
│   ├── 11_rn50_training_curves.png
│   ├── 12_rn50_confusion_matrix.png
│   ├── 13_rn50_roc_curve.png
│   ├── 14_model_comparison_bar.png
│   ├── 15_all_roc_curves.png
│   ├── 16_size_vs_accuracy.png
│   ├── 19_false_positives.png
│   ├── 20_false_negatives.png
│   ├── 21_confidence_distribution.png
│   └── 22_threshold_sensitivity.png
├── models/
│   ├── cnn_best.keras          (8.1 MB)
│   ├── cnn_final.keras         (8.1 MB)
│   ├── mobilenetv2_best.keras  (26.0 MB)
│   ├── mobilenetv2_final.keras (26.0 MB)
│   ├── resnet50_best.keras     (173.0 MB)
│   └── resnet50_final.keras    (173.0 MB)
└── reports/
    ├── cnn_classification_report.csv
    ├── mnv2_classification_report.csv
    ├── rn50_classification_report.csv
    ├── model_comparison.csv
    └── threshold_sensitivity.csv
```

---

## Running the Notebook

### On Kaggle (Recommended)

1. Open the notebook on Kaggle.
2. **Add the dataset:** right-side panel → **Add data** → search `surface-crack-detection` by `arunrk7` → **Add**.
3. **Enable GPU:** Session options → Accelerator → **GPU P100** or **T4 x2** → restart the session.
4. Click **Run All**.

The notebook auto-detects the Kaggle dataset path — no code changes needed.

| Model | Approx. GPU Training Time |
|-------|--------------------------|
| Custom CNN | ~10–15 min |
| MobileNetV2 | ~8–12 min |
| ResNet50 | ~10–15 min |
| **Full notebook end-to-end** | **~35–50 min** |

### Locally

```bash
# 1. Clone the repository
git clone https://github.com/saishivanandha10/Surface-Crack-Detection-using-CNN.git
cd Surface-Crack-Detection-using-CNN

# 2. Install dependencies
pip install tensorflow>=2.14 numpy pandas matplotlib seaborn \
            scikit-learn opencv-python-headless jupyter

# 3. Download the dataset from Kaggle and place it as:
#    ./surface-crack-detection/Positive/
#    ./surface-crack-detection/Negative/

# 4. Launch the notebook
jupyter notebook Surface_Crack_Detection_using_CNN.ipynb
```

For the full local and Kaggle setup guide, see [Instructions.md](Instructions.md).

---

## Requirements

| Package | Version |
|---------|---------|
| Python | 3.9+ |
| TensorFlow | ≥ 2.14 |
| NumPy | latest |
| Pandas | latest |
| Matplotlib | latest |
| Seaborn | latest |
| scikit-learn | latest |
| OpenCV | latest |

A CUDA-compatible GPU with ≥ 8 GB VRAM is strongly recommended for training.

---

## Reproducibility

All random seeds are fixed to `42` via `random.seed`, `numpy.random.seed`, and `tf.random.set_seed`. Data splits use `stratify=all_labels` and `random_state=42` throughout. Re-running the notebook from scratch on the same TensorFlow version produces identical splits, weights, and evaluation metrics.

---

## Notebook Sections

| # | Section | Description |
|---|---------|-------------|
| 0 | Warning Suppressor | Silences C++/CUDA TF runtime warnings before imports |
| 1 | Environment Setup | Imports, global constants, output directories |
| 2 | EDA | Image loading, class distribution, sample visualization |
| 3 | Preprocessing | 70/15/15 stratified split, `tf.data` pipelines, augmentation |
| 4 | Custom CNN | Architecture, training, evaluation (RQ1) |
| 5 | MobileNetV2 | 2-stage fine-tuning, evaluation (RQ1, RQ2) |
| 6 | ResNet50 | 2-stage fine-tuning, evaluation (RQ1, RQ2) |
| 7 | Model Comparison | Metrics table, bar charts, ROC curves, size vs. accuracy |
| 8 | Grad-CAM | Heatmap visualizations for CNN and MobileNetV2 (RQ4) |
| 9 | Error Analysis | False positives/negatives, confidence distributions, threshold sweep (RQ5) |
| 10 | Save & Summary | Model saving, figure inventory, final RQ1–RQ5 summary |

---

<div align="center">

Built with TensorFlow · Kaggle · Python

</div>
