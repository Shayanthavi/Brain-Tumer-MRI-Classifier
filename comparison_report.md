# Phase 13 – Conference Paper Comparison Report
# Automated Brain Tumor Detection and Classification using Deep Learning
# University of Ruhuna | EE7204 / EC7205

This report presents a comparative analysis between the deep learning methodology implemented in our project and the graph-based spatial prior approach described in the reference conference paper: *"Graph-Based Detection, Segmentation & Characterization of Brain Tumors: A Spatial Prior Approach for Low-Grade Gliomas"* by Sarah Parisot, Hugues Duffau, Stephane Chemouny, and Nikos Paragios.

---

## 1. Executive Summary

The reference conference paper proposes a hybrid machine learning and geometric approach to detect and segment Low-Grade Gliomas (LGGs) using a spatial prior (localized sparse graph) and a Gentle AdaBoost classifier with hand-crafted features. 

Our project implements an end-to-end deep learning workflow using four modern Convolutional Neural Network (CNN) architectures (Custom CNN, VGG16, ResNet50, and EfficientNetB0) to classify MRI scans into four distinct classes (Glioma, Meningioma, Pituitary Tumor, and No Tumor).

While the conference paper focuses on highly localized segmentation of a single tumor type (LGGs), our deep learning approach delivers a generalized, multi-class classification system that eliminates manual feature engineering and atlas registration dependencies, providing a simpler, more robust, and highly scalable solution.

---

## 2. Methodological Comparison

| Feature / Dimension | Conference Paper Approach | Our Deep Learning Approach |
| :--- | :--- | :--- |
| **Primary Objective** | Joint detection and segmentation of Low-Grade Gliomas (LGG). | 4-class classification (Glioma, Meningioma, Pituitary, No Tumor). |
| **Classifier / Model** | Gentle AdaBoost with a Graph-based Spatial Prior. | Custom CNN, VGG16, ResNet50, EfficientNetB0 with Softmax classifier. |
| **Feature Extraction** | Hand-crafted: Intensity (patch statistics), Texture (Gabor filters), and Symmetry (midline dissymmetry). | Automatically learned hierarchical convolutional feature maps. |
| **Input Preprocessing** | Rigid/Affine registration to a reference atlas, intensity normalization using Median and IQR. | OpenCV-based contour skull stripping, Gaussian/Median filtering, and Min-Max Normalization. |
| **Atlas Dependence** | **High**: Requires perfect alignment of the patient's MRI to a standard brain atlas. | **None**: Contour-based skull stripping isolates the brain without atlas registration. |
| **Classes Supported** | 1 class (LGG) vs Background. | 4 classes (Glioma, Meningioma, Pituitary, No Tumor). |
| **Dataset Size** | 113 MRI FLAIR scans. | 7,023 MRI scans (from Masoud Nickparvar's dataset). |
| **Segmentation output** | Voxel-wise probability map / Dice score. | Image-level classification with probability confidence scores. |

---

## 3. How Deep Learning Addresses the Limitations of the Conference Paper

The conference paper identifies several critical limitations in its own methodology (Section 11). Our deep learning implementation directly addresses and resolves these limitations:

### 1. Elimination of Hand-Crafted Feature Engineering
* **Conference Paper Limitation**: The performance of Gentle AdaBoost is heavily dependent on the quality of hand-crafted features (Gabor texture filters, intensity patches, and symmetry features). If the tumor does not display typical symmetry variance or has unusual texture, the classifier fails.
* **Our Solution**: Deep CNN backbones automatically learn optimal feature representations at multiple levels of abstraction (edges, textures, shapes, and semantic tumor characteristics) directly from the data during training, eliminating manual engineering.

### 2. Removal of Atlas Registration Dependency
* **Conference Paper Limitation**: The geometric graph prior relies entirely on accurate affine registration to a reference atlas. If the registration fails (due to anatomical variations or scanner differences), the spatial prior points to the wrong location, leading to misclassifications.
* **Our Solution**: Our pipeline uses a contour-based OpenCV skull stripping method that dynamically isolates the brain parenchyma by detecting the outer boundary. It is completely independent of external reference templates or atlases.

### 3. Handling of Large Tumors
* **Conference Paper Limitation**: The paper's spatial prior fails for very large tumors (>200 cubic cm). Large tumors deform the brain tissue significantly, spanning the domains of multiple graph clusters and confusing the localized prior.
* **Our Solution**: Convolutional Neural Networks look at global and local visual cues simultaneously using pooling and global average pooling layers. Because they do not rely on pre-defined spatial cluster coordinates, they easily classify large tumors based on their visual features.

### 4. Generalization Across Multiple Tumor Types
* **Conference Paper Limitation**: Specifically designed and tuned for Low-Grade Gliomas (LGGs) due to their predictable preferential locations. It cannot easily generalize to tumor types with random locations (e.g., glioblastomas, pituitary tumors, or meningiomas).
* **Our Solution**: Our transfer learning models generalize across multiple pathologies by training on a balanced dataset containing meningiomas (typically found on the brain surface), pituitary tumors (found at the base), and gliomas, showing excellent performance across all.

---

## 4. Deep Learning Architectures Comparison

In our implementation, we compare four distinct CNN architectures, each representing different design philosophies:

### 1. Custom CNN
* **Description**: A lightweight 4-layer convolutional neural network built from scratch.
* **Pros**: Small memory footprint, fast inference, trained specifically on the target MRI dataset.
* **Cons**: Limited capacity compared to transfer learning models; higher risk of overfitting on small datasets.

### 2. VGG16
* **Description**: A classic deep architecture using small $3 \times 3$ convolutional filters.
* **Pros**: Excellent feature extraction capabilities, simple sequential layout.
* **Cons**: Very large number of parameters (134M+ total), slow to train, and computationally expensive.

### 3. ResNet50
* **Description**: Introduces residual connections ("skip connections") to bypass layers.
* **Pros**: Prevents vanishing gradients, allowing deep training. Very stable convergence and high accuracy.
* **Cons**: Moderate size and memory usage during training.

### 4. EfficientNetB0
* **Description**: Scales depth, width, and resolution uniformly using a compound coefficient.
* **Pros**: State-of-the-art accuracy-to-efficiency ratio. Extremely lightweight (5.3M parameters) compared to ResNet50 (25.6M) and VGG16, making it ideal for clinical deployment.
* **Cons**: Slightly more complex architecture to fine-tune due to batch normalization statistics.

---

## 5. Conclusion

By shifting from the conference paper's hand-crafted features and geometric graph priors to end-to-end deep learning, we achieve a system that is:
1. **More General**: Can detect multiple tumor types and normal scans.
2. **More Robust**: Free from standard atlas registration errors.
3. **Simpler to Deploy**: Packaged as a clean end-to-end Python pipeline with a Streamlit interface.

This comparison demonstrates that while geometric priors are useful for small datasets (113 images), deep transfer learning is superior when scaled to modern datasets, providing high-fidelity automated classification for clinical decision support.
