# Surface Crack Detection Using CNN — Run Instructions

## Overview

This notebook implements binary image classification (Cracked vs. Negative) using three models:
- **Custom CNN** — a 4-block convolutional network trained from scratch
- **MobileNetV2** — transfer learning with 2-stage fine-tuning
- **ResNet50** — transfer learning with 2-stage fine-tuning

It also generates Grad-CAM visualizations, error analysis, and a full model comparison report.

---

## Requirements

### Python Version
Python 3.9 or higher is recommended.

### Hardware
- **CPU:** Supported, but training will be slow (Custom CNN: ~60–90 min; transfer learning models: ~30–60 min each).
- **GPU (recommended):** Any CUDA-compatible NVIDIA GPU with ≥8 GB VRAM. Training times drop to ~5–15 min per model.

### Install Dependencies

```bash
pip install tensorflow>=2.14 \
            numpy pandas matplotlib seaborn \
            scikit-learn opencv-python-headless \
            jupyter
```

> **Note on TensorFlow and CUDA:** If you have a GPU, install the matching `cuda` and `cudnn` packages for your system. TensorFlow ≥2.14 bundles CUDA support via pip — no separate installation is needed on most setups. The notebook suppresses known duplicate-plugin registration warnings automatically.

---

## Dataset Setup

1. Download the dataset from Kaggle:
   **[Surface Crack Detection — arunrk7](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection)**

2. Extract the archive. You should get a folder with the following structure:

   ```
   surface-crack-detection/
   ├── Positive/      ← 20,000 crack images (227×227 px, JPEG)
   └── Negative/      ← 20,000 non-crack images (227×227 px, JPEG)
   ```

3. Place the extracted folder either:
   - **Locally:** in the same directory as the notebook, named `surface-crack-detection/`
   - **On Kaggle:** the notebook auto-detects `/kaggle/input/datasets/arunrk7/surface-crack-detection`

   If you put it elsewhere, update this line in **Section 2A**:
   ```python
   LOCAL_PATH = Path('./surface-crack-detection')
   ```

---

## Running the Notebook

### Launch Jupyter

```bash
jupyter notebook Surface_Crack_Detection_using_CNN.ipynb
```

Or with JupyterLab:

```bash
jupyter lab Surface_Crack_Detection_using_CNN.ipynb
```

### Execution Order

Run all cells **top to bottom** in order. The sections must be executed sequentially because each one depends on variables and models defined earlier.

| Section | What it does |
|---------|-------------|
| **Cell 0** | Suppresses C++ TensorFlow warnings at the OS level before any imports |
| **1 — Setup** | Imports libraries, sets global constants, creates output directories |
| **2 — EDA** | Loads all image paths, plots class distribution and sample images |
| **3 — Preprocessing** | Splits data (70/15/15), builds `tf.data` pipelines with augmentation |
| **4 — Custom CNN** | Builds, trains, and evaluates the custom 4-block CNN |
| **5 — MobileNetV2** | 2-stage fine-tuning (feature extraction → fine-tune top 30 layers) |
| **6 — ResNet50** | 2-stage fine-tuning (feature extraction → fine-tune top 20 layers) |
| **7 — Comparison** | Aggregates metrics, bar charts, overlaid ROC curves, size vs. accuracy |
| **8 — Grad-CAM** | Generates heatmaps for Custom CNN and MobileNetV2 |
| **9 — Error Analysis** | False positive/negative visualization, confidence distributions, threshold sweep |
| **10 — Save** | Saves all models and prints the final research summary |

### Key Global Constants (Section 1)

You can adjust these before running:

| Constant | Default | Description |
|----------|---------|-------------|
| `IMG_SIZE` | `224` | Input resolution (must stay 224 for pretrained models) |
| `BATCH_SIZE` | `32` | Reduce to `16` if you run out of GPU memory |
| `EPOCHS_CNN` | `30` | Max epochs for the custom CNN (early stopping applies) |
| `EPOCHS_TL` | `20` | Max epochs for transfer learning Stage 2 |
| `LR` | `1e-3` | Initial learning rate |
| `SEED` | `42` | Random seed for reproducibility |

---

## Output Files

All outputs are written automatically to an `outputs/` directory created next to the notebook:

```
outputs/
├── figures/          ← 22 PNG plots (training curves, confusion matrices, Grad-CAMs, etc.)
├── models/           ← Saved Keras models (.keras format)
│   ├── cnn_best.keras
│   ├── cnn_final.keras
│   ├── mobilenetv2_best.keras
│   ├── mobilenetv2_final.keras
│   ├── resnet50_best.keras
│   └── resnet50_final.keras
└── reports/          ← CSV files
    ├── cnn_classification_report.csv
    ├── mnv2_classification_report.csv
    ├── rn50_classification_report.csv
    ├── model_comparison.csv
    └── threshold_sensitivity.csv
```

---

## Common Issues and Fixes

### `FileNotFoundError: Dataset not found`
The notebook cannot find the image folder. Check that `LOCAL_PATH` in Section 2A points to the correct path.

### `ResourceExhaustedError` / Out of GPU Memory
Reduce `BATCH_SIZE` from `32` to `16` in Section 1 and restart the kernel.

### TensorFlow C++ warnings on startup (cuDNN / cuBLAS)
These are cosmetic and do not affect results. Cell 0 suppresses them by redirecting `stderr` at the OS level before TF imports. If they still appear in your environment, you can safely ignore them.

### `UnimplementedError` related to `AdjustContrastv2`
The notebook already works around this by replacing `tf.image.random_contrast` with a manual equivalent. If this appears, ensure you are running Cell 0 before Section 3.

### Training is very slow without a GPU
Transfer learning models (MobileNetV2, ResNet50) can take 1–2 hours per model on CPU. Consider running on [Kaggle Notebooks](https://www.kaggle.com/) or [Google Colab](https://colab.research.google.com/) for free GPU access.

### Grad-CAM section fails on ResNet50
The ResNet50 Grad-CAM is not shown in this notebook by design — only Custom CNN (Section 8B) and MobileNetV2 (Section 8C) are visualized. ResNet50 uses the same `make_gradcam_heatmap` helper if you want to add it manually.

---

## Running on Kaggle

1. Upload the notebook to [kaggle.com/code](https://www.kaggle.com/code).
2. Add the dataset: **Dataset → Add → search "surface-crack-detection" by arunrk7**.
3. Enable GPU: **Session options → Accelerator → GPU P100**.
4. Click **Run All**. The notebook auto-detects the Kaggle path.

---

## Reproducibility

All random seeds are fixed to `42` via `random.seed`, `numpy.random.seed`, and `tf.random.set_seed`. The data split uses `stratify=all_labels` and `random_state=42` throughout. Re-running the notebook from scratch should produce the same splits, weights, and metrics, provided the same TensorFlow version is used.
