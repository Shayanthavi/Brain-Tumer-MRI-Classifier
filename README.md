# 🧠 Automated Brain Tumor Detection and Classification using Deep Learning

**University of Ruhuna · Department of Electrical and Information Engineering**  
**Module: Image Processing and Computer Vision (EE7204 / EC7205)**  
**Group 03 | Date: 2026**

---

## 1. Project Overview

This project implements an **Automated Computer-Aided Diagnosis (CAD)** system for detecting and classifying brain tumors from MRI images using Deep Learning. It trains and compares four CNN architectures:

| # | Model | Type |
|---|-------|------|
| 1 | Custom CNN | From scratch |
| 2 | VGG16 | Transfer Learning |
| 3 | ResNet50 | Transfer Learning |
| 4 | EfficientNetB0 | Transfer Learning |

The system includes a complete preprocessing pipeline (skull stripping, denoising, normalization), data augmentation, training, evaluation, and a **Streamlit-based web UI** for interactive MRI diagnosis.

---

## 2. Problem Statement

Brain tumors are life-threatening conditions requiring early detection. Manual MRI interpretation by radiologists is slow, prone to human error, and inconsistent across observers — especially for similar-looking tumor types such as Glioma, Meningioma, and Pituitary tumors.

This project develops a CNN-based CAD system to:
- Automatically detect brain tumors from MRI scans
- Classify them into one of four categories with high accuracy
- Provide a "second opinion" to assist radiologists

---

## 3. Objectives

1. Design and implement an automated system to detect brain tumors from MRI images using image processing techniques.
2. Classify detected tumors into: **Glioma**, **Meningioma**, **Pituitary**, and **No Tumor** using deep learning.
3. Preprocess MRI images using **noise removal**, **skull stripping**, and **normalization**.
4. Compare different CNN architectures (VGG16, ResNet50, EfficientNet) using accuracy, precision, and recall.
5. Develop a **user-friendly diagnostic interface** for medical professionals.

---

## 4. Dataset Information

| Property | Value |
|----------|-------|
| Source | Kaggle Brain Tumor MRI Dataset (masoudnickparvar) |
| Total Images | ~7,023 MRI images |
| Classes | Glioma, Meningioma, Pituitary, No Tumor |
| Format | JPEG/PNG, various sizes |
| Modality | T1, T2, FLAIR weighted MRI |

### Class Distribution

| Class | Approximate Count |
|-------|------------------|
| Glioma | ~1,621 |
| Meningioma | ~1,645 |
| No Tumor | ~2,000 |
| Pituitary | ~1,757 |

---

## 5. System Workflow

```
[Raw MRI Dataset]
       │
       ▼
[Download] → src/download.py
       │
       ▼
[Preprocessing] → src/preprocess.py
  ├── Grayscale → 3-channel
  ├── Resize 224×224
  ├── Gaussian Filter
  ├── Median Filter
  ├── Skull Stripping
  └── Min-Max Normalization
       │
       ▼
[Dataset Split] → 70% Train | 15% Val | 15% Test
       │
       ▼
[Data Augmentation] → src/augment.py (on training only)
  ├── Rotation ±30°
  ├── Horizontal Flip
  ├── Zoom ±15%
  ├── Shift ±10%
  ├── Brightness ±30%
  └── Contrast ±30%
       │
       ▼
[Model Training] → src/train.py
  ├── Custom CNN (single stage)
  ├── VGG16 (Stage 1: frozen backbone → Stage 2: fine-tune)
  ├── ResNet50 (Stage 1 → Stage 2)
  └── EfficientNetB0 (Stage 1 → Stage 2)
       │
       ▼
[Evaluation] → src/evaluate.py
  ├── Accuracy, Precision, Recall, F1-Score
  ├── Confusion Matrices
  ├── Training Curves
  └── Model Comparison Report
       │
       ▼
[Streamlit UI] → src/app.py
  ├── Upload MRI
  ├── Preview preprocessing stages
  ├── Predict class & confidence
  └── Interactive probability chart
```

---

## 6. Project Structure

```
project_com/
├── data/
│   ├── raw/                          # Downloaded raw MRI images
│   └── processed/
│       ├── train/{glioma, meningioma, notumor, pituitary}/
│       ├── val/  {glioma, meningioma, notumor, pituitary}/
│       └── test/ {glioma, meningioma, notumor, pituitary}/
├── models/                           # Saved Keras model weights
│   ├── custom_cnn_best.keras
│   ├── vgg16_best.keras
│   ├── resnet50_best.keras
│   ├── efficientnet_best.keras
│   └── *_history.json               # Training history logs
├── reports/                          # Evaluation outputs
│   ├── confusion_matrix_*.png
│   ├── training_curves_*.png
│   ├── model_comparison.png
│   └── model_comparison.csv
├── logs/                             # TensorBoard logs
├── src/
│   ├── __init__.py
│   ├── download.py                   # Phase 3: Dataset download
│   ├── preprocess.py                 # Phase 5: Preprocessing pipeline
│   ├── augment.py                    # Phase 6: Data augmentation
│   ├── models.py                     # Phase 7: Model architectures
│   ├── train.py                      # Phase 8: Training pipeline
│   ├── evaluate.py                   # Phase 9: Evaluation
│   └── app.py                        # Phase 10: Streamlit UI
├── run_pipeline.py                   # Master pipeline runner
├── requirements.txt                  # Python dependencies
├── compliance_report.md              # Proposal compliance report
├── comparison_report.md              # Conference paper comparison
└── README.md                         # This file
```

---

## 7. Dependency Installation

### Prerequisites

- Python 3.9 or 3.10 (recommended)
- pip package manager

### Step 1 – Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate
```

### Step 2 – Install All Dependencies

```bash
pip install -r requirements.txt
```

### Step 3 – Configure Kaggle API

1. Download an API token from your Kaggle account settings.
2. Place the file at:

```text
Windows: C:\Users\<your-user>\.kaggle\kaggle.json
Linux/macOS: ~/.kaggle/kaggle.json
```

3. Ensure the token file is readable by the Kaggle CLI.

### Step 4 – GPU Support (Optional)

For GPU-accelerated training, install the GPU version of TensorFlow:

```bash
pip install tensorflow[and-cuda]
```

Or install the CUDA toolkit separately and use:

```bash
pip install tensorflow-gpu
```

### Installed Dependencies Summary

| Package | Version | Purpose |
|---------|---------|---------|
| tensorflow | ≥2.12 | Deep learning framework |
| keras | ≥2.12 | Model building |
| opencv-python | ≥4.7 | Image preprocessing, skull stripping |
| scikit-learn | ≥1.2 | Train/test split, metrics |
| matplotlib | ≥3.7 | Plots, training curves |
| seaborn | ≥0.12 | Confusion matrix heatmaps |
| numpy | ≥1.23 | Array operations |
| Pillow | ≥9.5 | Image loading |
| streamlit | ≥1.24 | Web UI |
| tqdm | ≥4.65 | Progress bars |
| pandas | ≥2.0 | Results tabulation |
| plotly | ≥5.15 | Interactive UI charts |
| kaggle | ≥1.5.12 | Kaggle API dataset download |

---

## 8. Dataset Download Process

The dataset is automatically downloaded with the Kaggle API from the official Kaggle source:

```bash
python src/download.py
```

This will:
1. Authenticate with your Kaggle API token.
2. Download the official Kaggle Brain Tumor MRI Dataset.
3. Verify all four class folders exist.
4. Print a summary of the dataset.

**Before running this step**, place your `kaggle.json` token file in `C:\Users\<you>\.kaggle\kaggle.json` on Windows or `~/.kaggle/kaggle.json` on Linux/macOS. You can also download manually using:

```bash
pip install kaggle
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
```

---

## 9. Preprocessing Pipeline

Run preprocessing and dataset preparation:

```bash
python src/preprocess.py
```

This:
1. Scans raw images for all 4 classes.
2. Applies the full pipeline: Grayscale → Resize (224×224) → Gaussian Filter → Median Filter → Skull Strip → Normalize.
3. Splits the dataset: **70% train / 15% val / 15% test** (stratified by class).
4. Saves processed PNG images to `data/processed/`.

### Preprocessing Steps Explained

| Step | Method | Why |
|------|--------|-----|
| Grayscale → 3ch | Replicate channel ×3 | Pre-trained CNNs expect 3-channel input |
| Resize | `cv2.resize(224×224)` | Required by VGG16, ResNet50, EfficientNet |
| Gaussian Filter | `cv2.GaussianBlur(3×3)` | Smooth high-frequency scanner noise |
| Median Filter | `cv2.medianBlur(3×3)` | Remove impulse noise; preserve tumor edges |
| Skull Stripping | Otsu threshold + largest contour + crop | Remove non-brain tissue distracting the CNN |
| Normalization | Min-Max scaling to [0,1] | Stabilize training, prevent gradient issues |

---

## 10. Model Architectures

### Custom CNN

- 4 Convolutional blocks (32 → 64 → 128 → 256 filters)
- BatchNormalization + ReLU activation
- MaxPooling2D after each block
- Global Average Pooling → Dense(512) → Dropout(50%) → Softmax(4)
- ~2.5M parameters

### VGG16 Transfer Learning

- VGG16 backbone (ImageNet pre-trained, `include_top=False`)
- Custom head: GAP → Dense(512) → BN → Dropout(50%) → Dense(256) → Dropout(30%) → Softmax(4)
- Stage 1: frozen backbone, LR=1e-3
- Stage 2: last 4 layers unfrozen, LR=1e-5

### ResNet50 Transfer Learning

- ResNet50 backbone (ImageNet pre-trained)
- Same custom head as VGG16
- Stage 2: last 17 layers (final residual block) unfrozen

### EfficientNetB0 Transfer Learning

- EfficientNetB0 backbone (ImageNet pre-trained)
- Same custom head
- Stage 2: last 20 layers unfrozen
- Most parameter-efficient model

---

## 11. Training Instructions

### Train All Models

```bash
python src/train.py
```

### Train Specific Models

```bash
python src/train.py --models custom_cnn vgg16
```

### Run Full Pipeline

```bash
python run_pipeline.py
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss | Categorical Cross-Entropy |
| Batch Size | 32 |
| Stage 1 LR | 1e-3 |
| Stage 2 LR | 1e-5 |
| Max Epochs (Stage 1) | 1 |
| Max Epochs (Stage 2) | 1 |
| Early Stopping Patience | 5 epochs |

---

## 12. Evaluation Instructions

```bash
python src/evaluate.py
```

Outputs saved to `reports/`:
- `confusion_matrix_{model}.png` – per-model confusion matrices
- `training_curves_{model}.png` – accuracy and loss curves
- `model_comparison.png` – side-by-side bar chart
- `model_comparison.csv` – metrics table
- `classification_reports.json` – per-class detailed report

---

## 13. User Interface Usage

```bash
streamlit run src/app.py
```

Opens at: **http://localhost:8501**

**Interface Features:**
1. Select model from sidebar (Custom CNN, VGG16, ResNet50, EfficientNet)
2. Upload a brain MRI image (JPG/PNG)
3. View the full preprocessing pipeline (6 stages)
4. View the predicted class with a color-coded result badge
5. View confidence scores with an interactive bar chart
6. View per-class probability table and clinical information

---

## 14. Results

Verified results from the current smoke-run:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Custom CNN | 24.58% | 8.31% | 24.58% | 10.79% |

The current verified run is a baseline sanity-check on the processed dataset. The transfer learning models are implemented and ready to train next; they are expected to outperform the Custom CNN once full training is completed.

Expected results after full training (based on literature):

| Model | Expected Accuracy |
|-------|-----------------|
| Custom CNN | ~85–90% |
| VGG16 | ~90–93% |
| ResNet50 | ~92–96% |
| EfficientNetB0 | ~93–97% |

Actual results will vary based on available compute and random seed. See `reports/model_comparison.csv` after running evaluation.

---

## 15. Future Improvements

1. **Grad-CAM Visualisation**: Show which regions of the MRI the model focuses on (explainability).
2. **Segmentation**: Extend from classification to pixel-level tumour segmentation (U-Net).
3. **BraTS Dataset**: Incorporate multi-modal MRI (T1, T2, FLAIR, T1ce) for higher accuracy.
4. **Clinical Integration**: Add DICOM file support for direct hospital system integration.
5. **Ensemble Model**: Combine predictions from all 4 models for improved robustness.
6. **Federated Learning**: Train across multiple hospital datasets without sharing patient data.
7. **Mobile Deployment**: Convert to TensorFlow Lite for mobile device deployment.

---

## References

1. R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, Prentice Hall, 2008.
2. Z. N. K. Swati et al., "Brain tumor classification for MR images using transfer learning and fine-tuning," *Computerized Medical Imaging and Graphics*, 2019.
3. P. Afshar et al., "Brain tumor type classification using capsule networks," *IEEE ICIP*, 2018.
4. S. Pereira et al., "Brain tumor segmentation using CNNs in MRI images," *IEEE TMI*, 2016.
5. I. Abd El Kader et al., "Brain tumor detection and classification by a deep wavelet auto-encoder model," *Diagnostics*, 2021.
6. N. Abiwinanda et al., "Brain tumor classification using CNN," *World Congress Medical Physics*, 2019.
7. S. Deepak and P. M. Ameer, "Brain tumor classification using deep transfer learning," *Computers in Biology and Medicine*, 2019.
8. M. Sajjad et al., "Multi-grade brain tumor classification using deep CNN with extensive data augmentation," *Journal of Computational Science*, 2019.
9. M. A. Khan et al., "Brain tumor classification using deep learning: a review," *IEEE Access*, 2020.
10. M. M. Badza and M. C. Barjaktarovic, "Classification of brain tumors from MRI images using a CNN," *Applied Sciences*, 2020.
11. K. He et al., "Deep residual learning for image recognition," *CVPR*, 2016.
12. M. Tan and Q. Le, "EfficientNet: Rethinking model scaling for CNNs," *ICML*, 2019.
