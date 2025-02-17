# Mask Detection Using Machine Learning and Deep Learning Models

This repository showcases a **Computer Vision coursework** undertaken individually, exploring multiple approaches to detect whether individuals are wearing masks correctly, not wearing masks at all, or wearing them improperly. The project utilises **SIFT+SVM**, **HOG+MLP**, and a **CNN with Transfer Learning** to achieve robust mask detection performance.

## Project Background

Face mask detection has become a crucial aspect of public health. This coursework project investigates three core **machine learning and deep learning** approaches to classify images into:

1. **No Mask** (0)  
2. **Mask** (1)  
3. **Wearing Improperly** (2)

### Core Objectives
- Address **class imbalance** between mask and non-mask classes.
- Employ **augmentation** to mitigate limited dataset size.
- Compare **classical ML** pipelines vs. a **CNN-based** approach (ResNet-18).
- Identify which pipeline best balances accuracy, recall, and generalisation.

### Key Technologies & Libraries
- **Python** (OpenCV, Scikit-learn, PyTorch, TensorFlow)
- **Imbalanced-learn** for class rebalancing
- **Pretrained ResNet-18** for transfer learning

## Data Preprocessing

### Dataset Overview
- **Original Distribution**:
  - No Mask: 376 images
  - Mask: 1940 images
  - Wearing Improperly: 78 images

### Handling Class Imbalance
`imblearn` oversampling was performed to equalise each class to 1552 images, addressing severe imbalance.

### Image Enhancement & Augmentation
- **Contrast & Brightness** tweaks to improve feature clarity.
- **Augmentations** (random flips, rotations, colour jitter) applied selectively to the training set only.

## Implemented Methods

1. **SIFT + SVM**
   - **Feature Extraction**: Scale-Invariant Feature Transform (SIFT) to detect robust keypoints.
   - **Vocabulary**: Mini-batch KMeans for Bag-of-Words representation.
   - **Classification**: SVM with an RBF kernel.

2. **HOG + MLP**
   - **Feature Extraction**: Histogram of Oriented Gradients (HOG) captures edge/gradient features.
   - **Classifier**: Multi-Layer Perceptron (MLP) trained on the HOG feature set.

3. **CNN with Transfer Learning**
   - **Architecture**: ResNet-18 pretrained on ImageNet.
   - **Fine-tuning**: Last layer replaced for 3-class output, with targeted retraining.
   - **Optimisation**: Uses Stochastic Gradient Descent (SGD) and learning rate scheduling.

---

## Results & Insights

### SIFT+SVM
- **Overall Accuracy**: ~67%
- **Strength**: High precision (0.89) for the “Mask” class.
- **Weakness**: Low performance on “Wearing Improperly” (precision: 0.04, recall: 0.06).
- **Takeaway**: Effective for main class features but struggles with nuanced mask positions.

### HOG+MLP
- **Overall Accuracy**: ~85%
- **Strength**: Excellent recall (95%) for the “Mask” class.
- **Weakness**: Poor detection of “Wearing Improperly” (precision: 20%, recall: 6%).
- **Takeaway**: Performs well for dominant classes, but minority classes remain difficult.

### CNN with Transfer Learning
- **Validation Accuracy**: ~89.77% (peak around epoch 4).
- **Strength**: Balanced performance across classes, leveraging pretrained ImageNet weights.
- **Weakness**: Potential overfitting if not carefully regularised, higher computational needs.
- **Takeaway**: Best overall performance for complex mask scenarios.

**Overall Observations**  
- The CNN approach excelled in capturing subtle differences (e.g., wearing improperly).
- Classical ML methods (SIFT+SVM, HOG+MLP) performed well on majority classes but struggled with minority class generalisation.

---

## Challenges & Future Improvements
- **Minority Class Handling**:  
  The “Wearing Improperly” label remains challenging due to limited data. Potential enhancements include advanced oversampling (e.g., SMOTE variations) or synthetic data generation (GANs).  
- **Real-Time Deployment**:  
  Converting the CNN approach to an on-device or edge-based system for live detection.  
- **Ensemble Approaches**:  
  Combining classical ML predictions (SIFT+SVM, HOG+MLP) with the CNN to handle edge cases.  
- **Explainability**:  
  Tools like Grad-CAM could highlight which parts of an image the CNN focuses on, aiding in debugging and trust-building.

---

## User Guide & How to Run

1. **Clone the Repository**
    ```bash
    git clone https://github.com/alessandrosuarez/Computer-Vision-Pipeline-for-Face-Covering-Detection-FCD.git
    cd Computer-Vision-Pipeline-for-Face-Covering-Detection-FCD
    ```

2. **Install Dependencies**
   - Requires Python 3.8+ (3.6+ may also be compatible but not tested).
   - Install via pip:
     ```bash
     pip install -r requirements.txt
     ```
   - Or use conda:
     ```bash
     conda env create -f environment.yml
     conda activate mask-env
     ```

3. **Train & Evaluate Models**
   - **SIFT+SVM**: `Code/SIFTxSVM.ipynb`  
   - **HOG+MLP**: `Code/MLPxHOG.ipynb`  
   - **CNN**: `Code/CNN.ipynb`  
   Running each notebook will guide through data loading, training, and evaluation steps.

4. **Additional Testing**
   - `test_functions.ipynb`: Summaries of accuracy, precision, recall, F1-score, and confusion matrices for all models.
   - Optionally place personal images in `Personal_Dataset/test/` to verify model performance on external data.

---

## Repository Structure

```
Computer-Vision-Pipeline-for-Face-Covering-Detection-FCD
├─ CW_Dataset/
│   └─ (original mask dataset, sorted by classes)
├─ Code/
│   ├─ SIFTxSVM.ipynb
│   ├─ MLPxHOG.ipynb
│   └─ CNN.ipynb
├─ Models/
│   └─ (trained models in .pkl or .pth format)
├─ Personal_Dataset/
│   └─ test/
│       └─ (custom test images if needed)
├─ test_functions.ipynb
├─ environment.yml
├─ requirements.txt
└─ README.md
```

### Key Folders
- **CW_Dataset/**: Original project dataset.
- **Code/**: Jupyter notebooks for each learning pipeline.
- **Models/**: Trained model artefacts (SVM pickles, PyTorch .pth, etc.).
- **Personal_Dataset/**: Optional images for further testing.
- **test_functions.ipynb**: Final metrics and comparisons.
