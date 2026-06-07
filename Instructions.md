# Surface Crack Detection Using CNN - Kaggle Run Instructions

## Overview

This notebook implements binary image classification (Cracked vs. Negative) using three models:
- **Custom CNN** - a 4-block convolutional network trained from scratch
- **MobileNetV2** - transfer learning with 2-stage fine-tuning
- **ResNet50** - transfer learning with 2-stage fine-tuning

It also generates Grad-CAM visualizations, error analysis, and a full model comparison report.

---

## Step 1 - Add the Dataset

1. Open your Kaggle notebook.
2. In the right-side panel, click **Add data**.
3. Search for `surface-crack-detection` by `arunrk7`.
4. Click **Add**.

The dataset will be mounted automatically at:
```
/kaggle/input/datasets/arunrk7/surface-crack-detection/
├── Positive/      ← 20,000 crack images (227×227 px, JPEG)
└── Negative/      ← 20,000 non-crack images (227×227 px, JPEG)
```

No code changes are needed - the notebook detects this path automatically.

---

## Step 2 - Enable GPU Acceleration

Training all three models on CPU would take several hours. Enable a GPU before running:

1. Click **Session options** (top-right of the notebook editor).
2. Under **Accelerator**, select **GPU P100** or **GPU T4 x2**.
3. Click **Save** and restart the session.

> Do this **before** running any cells. Changing the accelerator mid-session requires a full restart.

---

## Step 3 - Run the Notebook

Click **Run All** or execute cells top to bottom in order. Do not skip or reorder sections - each one depends on variables and models defined earlier.

| Section | What it does |
|---------|-------------|
| **Cell 0** | Suppresses TensorFlow C++ runtime warnings before imports |
| **1 - Setup** | Imports libraries, sets global constants, creates output directories |
| **2 - EDA** | Loads all image paths, plots class distribution and sample images |
| **3 - Preprocessing** | Splits data 70/15/15, builds `tf.data` pipelines with augmentation |
| **4 - Custom CNN** | Builds, trains, and evaluates the custom 4-block CNN |
| **5 - MobileNetV2** | 2-stage fine-tuning (feature extraction → fine-tune top 30 layers) |
| **6 - ResNet50** | 2-stage fine-tuning (feature extraction → fine-tune top 20 layers) |
| **7 - Comparison** | Aggregates metrics, bar charts, overlaid ROC curves, size vs. accuracy |
| **8 - Grad-CAM** | Generates heatmaps for Custom CNN and MobileNetV2 |
| **9 - Error Analysis** | False positive/negative visualization, confidence distributions, threshold sweep |
| **10 - Save** | Saves all models and prints the final research summary |

### Approximate runtimes on Kaggle GPU (P100)

| Model | Training Time |
|-------|--------------|
| Custom CNN (up to 30 epochs) | ~10–15 min |
| MobileNetV2 (Stage 1 + 2) | ~8–12 min |
| ResNet50 (Stage 1 + 2) | ~10–15 min |
| Full notebook end-to-end | ~35–50 min |

---

## Step 4 - Save Outputs

Kaggle sessions are temporary. Before closing the notebook, save your outputs:

1. Go to the **Data** tab in the right-side panel.
2. Under **Output**, click **New dataset** to publish your `outputs/` folder.

Or download individual files directly from the file browser in the right-side panel.

### What gets generated

```
/kaggle/working/outputs/
├── figures/          ← 22 PNG plots (training curves, confusion matrices, Grad-CAMs, etc.)
├── models/
│   ├── cnn_best.keras
│   ├── cnn_final.keras
│   ├── mobilenetv2_best.keras
│   ├── mobilenetv2_final.keras
│   ├── resnet50_best.keras
│   └── resnet50_final.keras
└── reports/
    ├── cnn_classification_report.csv
    ├── mnv2_classification_report.csv
    ├── rn50_classification_report.csv
    ├── model_comparison.csv
    └── threshold_sensitivity.csv
```

---

## Key Constants (Section 1)

You can adjust these before running if needed:

| Constant | Default | When to change |
|----------|---------|----------------|
| `BATCH_SIZE` | `32` | Lower to `16` if you see an out-of-memory error |
| `EPOCHS_CNN` | `30` | Early stopping will cut this short automatically |
| `EPOCHS_TL` | `20` | Early stopping will cut this short automatically |
| `IMG_SIZE` | `224` | Do not change - required by MobileNetV2 and ResNet50 |

---

## Common Issues on Kaggle

### GPU not detected after enabling
Make sure you restarted the session after changing the accelerator. Check with:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```
This should return a non-empty list.

### Session disconnects mid-training
Kaggle sessions time out after ~9 hours of inactivity, but disconnects can happen earlier on free accounts. If this occurs, re-enable the GPU, rerun from Cell 0 through Section 1, then reload the best saved checkpoint:
```python
from tensorflow.keras.models import load_model
cnn_model = load_model('outputs/models/cnn_best.keras')
```

### Out-of-memory error during training
Reduce `BATCH_SIZE` to `16` in Section 1 and restart the session before re-running.

### TensorFlow C++ warnings (cuDNN / cuBLAS)
These are cosmetic and do not affect training or results. Cell 0 suppresses them automatically.
