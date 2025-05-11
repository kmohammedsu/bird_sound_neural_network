# Bird Sound Classification Using Deep Learning

## 1. Introduction

This project applies deep learning to classify bird species from audio recordings. By converting audio to spectrograms and training convolutional neural networks (CNNs), we demonstrate both binary and multi-class classification for ecological monitoring.

**Application domain:**  
- Bird sound classification for ecological monitoring and biodiversity research.

**Research questions:**  
- Can a CNN accurately classify bird species from spectrograms?
- What are the strengths and limitations of this approach compared to classical methods?

**Dataset:**  
- Preprocessed spectrograms generated from publicly available bird audio recordings.
- Each sample represents the time-frequency content of a bird call.
- The dataset includes twelve bird species, with imbalanced representation (37–630 samples per species).
- Data is provided as `.hdf5` files and raw `.mp3` files for inference.

---

## 2. File Structure

```
bird_sound_neural_network/
├── data/
│ ├── external_test_clips/
│ │ ├── test1.mp3
│ │ ├── test2.mp3
│ │ └── test3.mp3
│ ├── bird_spectrogram.hdf5
│ ├── bird_spectrogram_splits.hdf5
│ └── external_test_processed.hdf5
├── models/
│ ├── binary_norfil_houfin_cnn_small_model.h5
│ ├── multi_class.h5
│ └── multiclass.weights.h5
├── notebooks/
│ ├── 01_binary_classification.ipynb
│ ├── 02_multiclass_classification.ipynb
│ └── 03_external_test_data.ipynb
├── requirements.txt
└── README.md
```

---

## 3. Theoretical Background

### Convolutional Neural Networks (CNNs)
CNNs are deep learning models designed for grid-like data such as images and spectrograms. They use convolutional layers to detect local patterns (e.g., frequency contours in bird calls), pooling layers for dimensionality reduction, and fully connected layers for classification. ReLU activation introduces non-linearity, and dropout helps prevent overfitting.

### Binary vs. Multiclass Classification
- **Binary Classification:** Distinguishes between two classes (e.g., Northern Flicker vs. House Finch).
- **Multiclass Classification:** Identifies one species from twelve options using a softmax output.

### MP3-to-Spectrogram Conversion and Data Preprocessing
- **Subsampling:** All audio is resampled to 22050 Hz.
- **Clipping:** The first 3 seconds of each clip are selected. Optionally, users can manually select a 3-second window where the bird is audible.
- **Spectrogram Generation:** Each 2-second window is transformed into a spectrogram (128 frequency bins × 517 time steps) using the Short-Time Fourier Transform (STFT). Amplitude is converted to decibels.
- **Saving:** All bird calls for all clips in a species are saved individually, resulting in an uneven number of samples per species (37–630).
- **Resizing and Normalization:** Spectrograms are resized to 343×256 for model input and normalized to [-1, 1].
- **Storage:** Processed data is stored in HDF5 format.

---

## 4. Methodology

- **Data Processing:**  
  - All MP3 files are subsampled to 22050 Hz and clipped to 3 seconds.
  - Spectrograms are generated for each 2-second window, converted to decibel scale, resized, and normalized.
  - Labels are one-hot encoded for multiclass tasks.

- **Model Implementation:**  
  - Binary model: 3 conv layers (8→16→32 filters), dense layer, sigmoid output.
  - Multiclass model: 3 conv layers (32→64→128 filters), dense layer, softmax output.
  - Training: Adam optimizer, binary/categorical cross-entropy loss, 20 epochs.

- **Testing & Validation:**  
  - Data split into training, validation, and test sets.
  - Hyperparameters tuned via validation performance.
  - Evaluation metrics: accuracy, MAE, confusion matrix, classification report.

- **External Testing:**  
  The multiclass model (`multiclass.weights.h5`) was used for inference on new MP3 files in `external_test_clips/`, after processing them identically to training data.

---

## 5. Results

### Binary Classification
- **Test accuracy:** 100%
- **Perfect separation** of Northern Flicker and House Finch

### Multiclass Classification
- **Test accuracy:** 71.9%
- **Test MAE:** 0.068

#### Confusion Matrix (rows: true, columns: predicted)
|           | amecro | amerob | bewwre | bkcchi | daejun | houfin | houspa | norfli | rewbla | sonspa | spotow | whcspa |
|-----------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| **amecro** | 17     | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 2      |
| **amerob** | 0      | 18     | 0      | 0      | 0      | 0      | 0      | 0      | 1      | 0      | 0      | 0      |
| **bewwre** | 0      | 0      | 19     | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      |
| **bkcchi** | 0      | 0      | 0      | 14     | 0      | 0      | 0      | 6      | 0      | 0      | 0      | 0      |
| **daejun** | 0      | 0      | 3      | 0      | 11     | 0      | 0      | 0      | 0      | 0      | 5      | 0      |
| **houfin** | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 19     |
| **houspa** | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 20     | 0      | 0      |
| **norfli** | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 20     | 0      | 0      | 0      | 0      |
| **rewbla** | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 19     | 0      | 0      | 0      |
| **sonspa** | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 19     | 0      | 0      |
| **spotow** | 0      | 0      | 9      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 10     | 0      |
| **whcspa** | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 19     |

- **Most challenging species:**  
  - House Finch (`houfin`): all misclassified as White-crowned Sparrow (`whcspa`)
  - House Sparrow (`houspa`): all misclassified as Song Sparrow (`sonspa`)
  - Spotted Towhee (`spotow`): 9/19 misclassified as Bewick’s Wren (`bewwre`)
  - Black-capped Chickadee (`bkcchi`): 6/20 misclassified as Northern Flicker (`norfli`)
### Species Label Key

| Code    | Species                  | Code    | Species                  |
|---------|--------------------------|---------|--------------------------|
| amecro  | American Crow            | houfin  | House Finch              |
| amerob  | American Robin           | houspa  | House Sparrow            |
| bewwre  | Bewick's Wren            | norfli  | Northern Flicker         |
| bkcchi  | Black-capped Chickadee   | rewbla  | Red-winged Blackbird     |
| daejun  | Dark-eyed Junco          | sonspa  | Song Sparrow             |
| spotow  | Spotted Towhee           | whcspa  | White-crowned Sparrow    |

### Inference on External Test Data
The multiclass model was used to classify new MP3 files:

- **test1.mp3** → `houspa` (confidence: **0.83**)
- **test2.mp3** → `houspa` (confidence: **0.78**)
- **test3.mp3** → `houspa` (confidence: **0.85**)
All were identified as House Sparrow with high confidence.

---

## 6. Discussion

**Limitations:**
- **Class imbalance** led to bias toward overrepresented species.
- **Training time:** Multiclass CNN required ~45 seconds per epoch (total ~15 minutes); binary model trained in ~1 minute.
- **Generalization:** Model may struggle with noisy or field recordings and unseen species.
- **No data augmentation** was used, limiting robustness.

**Challenging Species and Confusions:**
- House Finch and House Sparrow were frequently confused with other sparrow species, likely due to similar frequency patterns and call structures.
- Spotted Towhee and Bewick’s Wren exhibited confusion, possibly due to overlapping harmonic content.
- Listening to the calls and inspecting spectrograms confirmed these species have visually and acoustically similar features.

**Alternative Models:**
- Classical approaches (Random Forests, SVMs) could be used with hand-crafted features, but may miss subtle patterns.
- Hybrid models (CNN+RNN) could capture temporal dependencies.
- **Why neural networks?**  
  CNNs automatically learn hierarchical features from spectrograms, excel at spatial pattern recognition, and scale well to multiclass tasks.

---

## 7. Conclusions

This work demonstrates that CNNs can reliably classify bird species from audio recordings, achieving 72% accuracy on a 12-class problem. The pipeline is adaptable to other bioacoustic tasks and provides a foundation for automated wildlife monitoring systems. Future improvements could include data augmentation, advanced architectures, and addressing class imbalance.

---
## 8.Replication Guide

Follow these steps to replicate the project on either Windows or macOS.

---

### 1. Environment Setup

#### Windows

1. **Install Python**
   - Recommended: Install Python via [Microsoft Store](https://apps.microsoft.com/detail/python-3.12/9NJ46SX7X90P)
   - Or download from [python.org](https://www.python.org/downloads/)

2. **Create and Activate Virtual Environment**
```
python -m venv birdenv
.\birdenv\Scripts\activate
```


#### macOS

1. **Install Python**
- Recommended: Use Homebrew
  ```
  brew install python@3.12
  ```
- Or download from [python.org](https://www.python.org/downloads/)

2. **Create and Activate Virtual Environment**
```
python3 -m venv birdenv
source birdenv/bin/activate
```

---

### 2. Install Dependencies

Install all required Python packages:
```
pip install -r requirements.txt
```


---

### 3. Run Data Preprocessing

Navigate to the `scripts` directory and run the preprocessing script.  
Ensure your MP3 files are in the appropriate data folder (`data/external_test_clips/`).
```
cd scripts
python data_preprocessing.py
```

---

### 4. Execute Notebooks in Order

1. **Binary Classification**  
   Open and run:
```
notebooks/01_binary_classification.ipynb
```

- Trains model for Northern Flicker vs. House Finch.

2. **Multiclass Classification**  
Open and run:
```
notebooks/02_multiclass_classification.ipynb
```
- Trains model for all 12 bird species.

3. **External Test Inference**  
Open and run:
```
notebooks/03_external_test_data.ipynb
```

- Predicts species for new MP3 files in `data/external_test_clips/`.





---

### Verification

1. Confirm Python version:
```
python --version # Should be 3.10 or higher
```
2. Check critical packages:
```
pip show librosa tensorflow h5py
```

---


## 9. References

1. **Spectrogram Conversion Code**:  
   [Prof. Mendible's GitHub](https://github.com/mendible/5322/tree/main/Homework%203)  
   *(Used for MP3-to-spectrogram preprocessing)*


2. **Librosa Documentation**:  
   [librosa.org/doc](https://librosa.org/doc/)  
   *(Audio loading, STFT, and spectrogram generation)*

3. **TensorFlow Documentation**:  
   [tensorflow.org/api_docs](https://www.tensorflow.org/api_docs)  
   *(CNN implementation and training)*

---

