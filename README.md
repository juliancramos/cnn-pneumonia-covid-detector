# Chest X-Ray Classification: Normal vs. Pneumonia vs. COVID-19

Implementation of a custom **Convolutional Neural Network (CNN)** for multi-class classification of chest X-ray images into three categories: **Normal**, **COVID-19**, and **Pneumonia**. The project addresses a severely imbalanced dataset by implementing and comparing three balancing strategies—class weighting, undersampling, and data augmentation—and evaluates their impact on classification metrics including accuracy, precision, recall, and F1-score.

**Authors:** Sergio Ortíz · Julián Ramos · Melissa Ruiz  
**Course:** Machine Learning Techniques  
**Date:** February 2026

---

## Table of Contents

1. [Project Purpose & Context](#1-project-purpose--context)
2. [Data Pipeline & Preprocessing](#2-data-pipeline--preprocessing)
3. [Handling Class Imbalance: The 3 Experiments](#3-handling-class-imbalance-the-3-experiments)
4. [CNN Architecture](#4-cnn-architecture)
5. [Conclusion & Future Work](#5-conclusion--future-work)
6. [Learning Curves & Confusion Matrices](#6-learning-curves--confusion-matrices)
7. [Tech Stack](#7-tech-stack)
8. [Setup & Installation](#8-setup--installation)
9. [Repository Structure](#9-repository-structure)

---

## 1. Project Purpose & Context

This project builds a custom **Convolutional Neural Network (CNN)** to classify chest X-ray images into three mutually exclusive categories:

| Category     | Description                                  |
|:-------------|:---------------------------------------------|
| **NORMAL**   | No pathological findings in the lung fields  |
| **COVID-19** | Radiological patterns consistent with SARS-CoV-2 infection |
| **PNEUMONIA**| Opacities consistent with bacterial or viral pneumonia |

The primary objective is to design, train, and evaluate a CNN capable of distinguishing between these three classes from raw X-ray images. A key challenge in this task is the **severe class imbalance** in the dataset (NORMAL ~56%, Pneumonia ~27%, COVID-19 ~17%), which can bias the model toward the majority class if left unaddressed. Three different balancing strategies are implemented and compared to analyze their effect on model performance.

### Note on Evaluation Metrics

Because the dataset is imbalanced and the classification involves medical categories, overall accuracy alone is insufficient to evaluate model quality. The project uses **Precision**, **Recall**, **F1-Score**, and **Confusion Matrices** as complementary metrics. In this domain, a **False Negative** (a disease case classified as Normal) and a **False Positive** (a healthy case classified as disease) carry different implications. The analysis of each strategy documents this distinction to provide context for the metric trade-offs observed.

---

## 2. Data Pipeline & Preprocessing

### 2.1 Data Sources

The dataset is assembled from two public Kaggle repositories:

| Dataset | Provider | Classes Extracted |
|:--------|:---------|:------------------|
| Chest X-Ray Images (Pneumonia) | Paul Mooney | `NORMAL`, `PNEUMONIA` |
| COVID-19 Radiography Database | Tawsifur Rahman | `COVID-19`, `NORMAL`, `Viral Pneumonia` → merged into `PNEUMONIA` |

The `Lung_Opacity` folder from the COVID-19 Radiography Database is explicitly **excluded** because it is a general radiological label (potentially covering pneumonia, edema, cancer, etc.) that would introduce class ambiguity. The `Viral Pneumonia` folder (1,345 images) is merged into the general `PNEUMONIA` class to increase sample diversity.

### 2.2 Unified DataFrame

All image paths and labels from both sources are aggregated into a single Pandas DataFrame. The original train/test/val splits from the Pneumonia dataset are ignored; all images are pooled into one collection for a custom split later. The DataFrame is then shuffled (`sample(frac=1, random_state=42)`) to prevent the model from learning class order.

### 2.3 Dataset Composition

| Class         | Samples | Proportion |
|:--------------|--------:|-----------:|
| **NORMAL**    | 11,775  | 56.0%      |
| **PNEUMONIA** |  5,618  | 26.7%      |
| **COVID-19**  |  3,616  | 17.2%      |
| **Total**     | 21,009  | 100%       |

The distribution shows a clear **class imbalance**: `NORMAL` exceeds both pathological classes combined, which can bias the model toward predicting the majority class.

### 2.4 Image Resizing

All images are standardized to **224 × 224 pixels** before being processed by the neural network.

**Rationale:**
- **Architecture compatibility:** 224×224 is the standard input size for established CNN architectures (VGG16, ResNet, DenseNet).
- **Computational efficiency:** Smaller input dimensions reduce GPU memory consumption and training time compared to higher resolutions (e.g., 512×512).
- **Feature retention:** The pathological patterns relevant to Pneumonia and COVID-19 detection (diffuse opacities, ground-glass patterns) are macro-scale features that remain identifiable at 224×224 resolution.

Resizing is performed via OpenCV's `cv2.resize()` during preprocessing and via TensorFlow's `image_dataset_from_directory(image_size=(224, 224))` during dataset loading.

### 2.5 Pixel Normalization

Pixel intensities are scaled from the integer range `[0, 255]` to the floating-point range `[0.0, 1.0]` by dividing by 255.

This normalization is implemented in two ways within the pipeline:
1. **Verification stage:** Manual division using `img.astype('float32') / 255.0` to generate before/after histograms confirming distribution preservation.
2. **Model inference:** A `Rescaling(1./255)` layer is embedded as the first layer of the CNN, ensuring that normalization is applied consistently to every input during both training and inference.

**Benefits:**
- **Numerical stability:** Prevents excessively large gradient values during backpropagation.
- **Training efficiency:** Maintains a consistent numerical scale that facilitates smoother optimization.

---

## 3. Handling Class Imbalance: The 3 Experiments

Three independent strategies are evaluated to address the class distribution imbalance identified in the EDA. All experiments use the **same CNN architecture** (Section 4), **same optimizer** (Adam), **same loss function** (Sparse Categorical Cross-Entropy), and **same number of epochs** (15) to ensure that any performance differences are attributable solely to the data-handling strategy.

An 80/20 train–validation split is applied using `tf.keras.utils.image_dataset_from_directory()` with a fixed seed (`42`) for reproducibility. The TensorFlow input pipeline is optimized with `.prefetch(buffer_size=tf.data.AUTOTUNE)`.

---

### 3.1 Strategy A — Class Weighting

**Method:** The dataset remains unchanged (21,009 images). Weights are injected into the loss function via `model.fit(class_weight=weights_dict)` so that misclassifications of minority classes incur proportionally higher penalties.

**Weight Calculation:** Using `sklearn.utils.class_weight.compute_class_weight('balanced')`:

$$W_j = \frac{N_{total}}{N_{classes} \times N_j}$$

| Class       | Samples | Computed Weight |
|:------------|--------:|----------------:|
| COVID-19    |   3,616 |          1.9364 |
| NORMAL      |  11,775 |          0.5943 |
| PNEUMONIA   |   5,618 |          1.2466 |

**Results:**

| Metric                  | Value    |
|:------------------------|:---------|
| Overall Accuracy        | ~93%     |
| COVID-19 Recall         | 0.98     |
| PNEUMONIA Recall        | 0.96     |
| COVID-19 Precision      | 0.79     |
| NORMAL Recall           | 0.89     |
| False Negatives (COVID) | 13 / 713 |
| False Positives (COVID) | 174 NORMAL → COVID-19 |

**Analysis:** The class weighting strategy maximizes recall for disease classes (COVID-19: 0.98, Pneumonia: 0.96), ensuring that nearly all positive cases are detected. The cost is a reduction in precision: 174 Normal images are incorrectly flagged as COVID-19, resulting in a COVID-19 precision of 0.79. Training curves show observable fluctuations in validation metrics, consistent with the amplified influence of minority class errors on the weighted loss.

---

### 3.2 Strategy B — Undersampling

**Method:** The `NORMAL` class is randomly downsampled from 11,775 to 5,618 samples to match `PNEUMONIA`. `COVID-19` and `PNEUMONIA` samples are preserved. The resulting dataset contains 14,852 images.

> A stricter 1:1:1 balance (reducing all classes to 3,616) was considered but rejected because it would require discarding approximately 2,000 pathological Pneumonia images, which are valuable for learning disease features.

**Results:**

| Metric                  | Value    |
|:------------------------|:---------|
| Overall Accuracy        | ~93%     |
| COVID-19 Recall         | 0.81     |
| NORMAL Recall           | 0.97     |
| COVID-19 Precision      | 0.96     |
| False Negatives (COVID) | 118 COVID-19 → NORMAL |

**Analysis:** The undersampling strategy produces stable, smooth learning curves with close alignment between training and validation metrics. Precision for COVID-19 increases to 0.96 and NORMAL recall reaches 0.97, indicating fewer false positives. However, COVID-19 recall drops to 0.81, meaning approximately 20% of positive COVID-19 cases go undetected—118 cases misclassified as NORMAL. In a clinical screening scenario, this false negative rate is not acceptable.

---

### 3.3 Strategy C — Offline Data Augmentation

**Method:** Synthetic images are generated for `COVID-19` and `PNEUMONIA` until each class reaches 11,775 samples, matching `NORMAL`. Augmentation operations include:

- **Random zoom:** Scale factor between 0.85 and 0.95, implemented as a random crop followed by resize to 224×224.
- **Brightness/contrast adjustment:** Contrast (α) uniformly sampled from [0.85, 1.15], brightness (β) uniformly sampled from [-15, +15].

Operations such as horizontal flips and large rotations are **excluded** to preserve anatomical validity (e.g., heart laterality).

Augmented images are physically saved to disk in the `dataset_augmented_generated/` directory using unique UUID-based filenames. The final DataFrame (`df_augmented`) contains 35,325 entries (21,009 original + 14,316 synthetic).

**Results:**

| Metric           | Reported Value |
|:-----------------|:---------------|
| Overall Accuracy | ~97%           |
| COVID-19 Recall  | 0.95           |
| PNEUMONIA Recall | 0.99           |
| NORMAL Recall    | 0.95           |

> [!CAUTION]
> **Data Leakage Identified:** The augmentation was applied to the **entire dataset before** the train/validation split. As a result, transformed versions of validation images were present in the training set. The model was exposed to augmented variants of its own validation samples during training, which explains the inflated performance metrics. **The reported 97% accuracy does not reflect true generalization to unseen data and is therefore invalid for clinical assessment.**

**Root Cause:** The `df_augmented` DataFrame—containing both original and synthetic images—was passed as a single block to `image_dataset_from_directory()`, which then performed the 80/20 split. Since multiple augmented copies of the same source image could appear in both partitions, the validation set was contaminated.

---

## 4. CNN Architecture

The network follows a sequential structure from feature extraction to classification, implemented using `tf.keras.models.Sequential`.

### Architecture Overview

```
Input (224 × 224 × 3)
    │
    ▼
Rescaling (1/255)               ─── Normalization: [0,255] → [0,1]
    │
    ▼
Conv2D(32, 3×3, ReLU)          ─── 896 params  → (222, 222, 32)
MaxPooling2D(2×2)               ─── 0 params    → (111, 111, 32)
    │
    ▼
Conv2D(32, 3×3, ReLU)          ─── 9,248 params → (109, 109, 32)
MaxPooling2D(2×2)               ─── 0 params    → (54, 54, 32)
    │
    ▼
Conv2D(64, 3×3, ReLU)          ─── 18,496 params → (52, 52, 64)
MaxPooling2D(2×2)               ─── 0 params     → (26, 26, 64)
    │
    ▼
Conv2D(64, 3×3, ReLU)          ─── 36,928 params → (24, 24, 64)
MaxPooling2D(2×2)               ─── 0 params     → (12, 12, 64)
    │
    ▼
Dropout(0.2)                    ─── Regularization
GlobalAveragePooling2D          ─── 0 params     → (64,)
    │
    ▼
Dense(32, ReLU)                 ─── 2,080 params
Dropout(0.2)                    ─── Regularization
    │
    ▼
Dense(3, Softmax)               ─── 99 params   → Probability over 3 classes
```

**Total trainable parameters:** 67,747

### Design Decisions

| Component | Choice | Rationale |
|:----------|:-------|:----------|
| **Kernel size** | 3×3 | Standard minimal kernel; captures local spatial patterns |
| **Filter progression** | 32 → 32 → 64 → 64 | Gradual increase captures progressively complex features |
| **Activation** | ReLU | Mitigates vanishing gradient; computationally efficient |
| **Pooling** | 2×2 Max Pooling | Reduces spatial dimensions; provides translation invariance |
| **Global Average Pooling** | Instead of Flatten | Reduces parameter count; improves generalization |
| **Dropout** | 20% (two layers) | Regularization to reduce overfitting |
| **Output activation** | Softmax | Produces probability distribution for mutually exclusive classes |

### Optimizer & Loss Function

- **Optimizer:** Adam — combines Momentum (exponentially weighted average of past gradients for faster convergence) and RMSProp (per-parameter adaptive learning rates).
- **Loss function:** Sparse Categorical Cross-Entropy — suited for multi-class problems with integer labels, more memory-efficient than standard Categorical Cross-Entropy (no one-hot encoding required).
- **Epochs:** 15 per experiment.
- **Batch size:** 32.

---

## 5. Conclusion & Future Work

### Final Strategy Selection

Based on the experimental results, **Strategy A (Class Weighting)** is selected as the preferred configuration.

| Strategy | Accuracy | COVID-19 Recall | COVID-19 Precision | Observation |
|:---------|:--------:|:---------------:|:------------------:|:------------|
| A — Class Weighting   | ~93% | **0.98** | 0.79 | High recall, lower precision |
| B — Undersampling     | ~93% | 0.81     | 0.96 | High precision, lower recall |
| C — Data Augmentation | ~97% | 0.95     | —    | **Invalidated** (data leakage) |

Strategy A achieves the highest recall for disease classes (COVID-19: 0.98, Pneumonia: 0.96), meaning 13 out of 713 COVID-19 cases were missed. The trade-off is a lower precision: 174 Normal images were classified as COVID-19. In a medical context, False Negatives (missed disease) carry different implications than False Positives (healthy cases flagged for review), which is a relevant consideration when selecting between strategies.

Strategy B produces the most stable learning curves and higher precision, but COVID-19 recall drops to 0.81, meaning approximately 20% of positive cases go undetected.

Strategy C's metrics are invalidated due to data leakage and cannot be used for comparison.

### Future Work

1. **Fix the augmentation pipeline:** Apply data augmentation **exclusively to the training subset** after performing the train/validation split. This eliminates the data leakage and enables a valid assessment of augmentation as a balancing strategy.
2. **Transfer learning:** Evaluate pre-trained architectures (e.g., VGG16, ResNet50, DenseNet121) on ImageNet weights as feature extractors, potentially improving performance without requiring a large custom dataset.
3. **Hyperparameter optimization:** Systematic tuning of learning rate, dropout rate, and network depth using grid search or Bayesian optimization.
4. **External validation:** Test the selected model on an independent, unseen dataset from a different clinical source to estimate real-world generalization.
5. **Artifact mitigation:** Investigate segmentation-based preprocessing to mask medical annotations (e.g., "R", "L" markers, ECG leads) identified during visual inspection, which could introduce bias.

---

## 6. Learning Curves & Confusion Matrices

<!-- Insert the training/validation accuracy and loss curves for each strategy below -->

### Strategy A — Class Weighting

<!-- ![Strategy A: Learning Curves](images/strategy_a_learning_curves.png) -->
<!-- ![Strategy A: Confusion Matrix](images/strategy_a_confusion_matrix.png) -->

### Strategy B — Undersampling

<!-- ![Strategy B: Learning Curves](images/strategy_b_learning_curves.png) -->
<!-- ![Strategy B: Confusion Matrix](images/strategy_b_confusion_matrix.png) -->

### Strategy C — Data Augmentation

<!-- ![Strategy C: Learning Curves](images/strategy_c_learning_curves.png) -->
<!-- ![Strategy C: Confusion Matrix](images/strategy_c_confusion_matrix.png) -->

> [!NOTE]
> Export the plots generated during notebook execution and place them in an `images/` directory at the repository root. Update the paths above to reference the saved files.

---

## 7. Tech Stack

| Category            | Technology                    | Purpose                                              |
|:--------------------|:------------------------------|:-----------------------------------------------------|
| Language            | Python 3                      | Core implementation language                         |
| Deep Learning       | TensorFlow / Keras            | CNN construction, training, and evaluation           |
| Data Manipulation   | Pandas, NumPy                 | DataFrame operations, numerical computations         |
| Image Processing    | OpenCV (`cv2`)                | Image reading, resizing, color space conversion      |
| Visualization       | Matplotlib, Seaborn           | Training curves, confusion matrices, distribution plots |
| ML Utilities        | Scikit-learn                  | Class weight computation, classification report, confusion matrix |
| Environment         | Google Colab                  | Execution environment with GPU support               |
| Data Source         | Kaggle API                    | Automated dataset download and extraction            |
| Reproducibility     | Fixed seed (`SEED = 42`)      | Consistent results across Python, NumPy, TensorFlow  |

---

## 8. Setup & Installation

### Prerequisites

- Python 3.8+
- A Kaggle account with API credentials (`kaggle.json`)
- Google Colab (recommended) or a local environment with GPU support

### Running on Google Colab (Recommended)

1. Upload `kaggle.json` to the root of your Google Drive (`/MyDrive/kaggle.json`).
2. Open the notebook `cnn-pneumonia-covid-detector.ipynb` in [Google Colab](https://colab.research.google.com/).
3. Run all cells sequentially. The notebook will:
   - Mount Google Drive and configure the Kaggle API.
   - Download and extract both datasets automatically.
   - Execute the full pipeline: EDA → Preprocessing → Training → Evaluation.

### Running Locally

```bash
# 1. Clone the repository
git clone https://github.com/juliancramos/cnn-pneumonia-covid-detector.git
cd cnn-pneumonia-covid-detector

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install tensorflow pandas numpy matplotlib seaborn opencv-python scikit-learn tqdm kaggle

# 4. Configure Kaggle API
mkdir -p ~/.kaggle
cp /path/to/your/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 5. Download datasets
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia --unzip
kaggle datasets download -d tawsifurrahman/covid19-radiography-database --unzip

# 6. Launch Jupyter and open the notebook
jupyter notebook cnn-pneumonia-covid-detector.ipynb
```

> [!IMPORTANT]
> The notebook contains Google Colab-specific commands (`drive.mount()`, `!kaggle`, `!unzip`). If running locally, replace these with their equivalent system commands or execute the Kaggle CLI directly in your terminal as shown above.

---

## 9. Repository Structure

```
cnn-pneumonia-covid-detector/
├── cnn-pneumonia-covid-detector.ipynb   # Jupyter Notebook (primary executable)
├── cnn-pneumonia-covid-detector.py      # Auto-generated Python script from Colab
├── README.md                            # This file
└── images/                              # (To be created) Exported plots and figures
```

---

<p align="center">
  Developed as part of the Machine Learning Techniques course — Seventh Semester, February 2026.
</p>