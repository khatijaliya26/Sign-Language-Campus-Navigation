# Sign Language-Based Campus Navigation System Using Deep Learning

A deep learning-based gesture recognition system designed to support campus navigation by identifying predefined sign language gestures and mapping them to corresponding campus locations and navigation instructions.

## Project Overview

This project explores the use of computer vision and deep learning to develop a sign-based campus navigation system. It recognizes a predefined set of nine dynamic hand gestures from video input and maps each recognized gesture to a corresponding campus location.

The system was developed in two major phases:

1. **Landmark-Based Gesture Recognition:** Uses MediaPipe to extract hand landmarks, followed by BiLSTM and CNN–BiLSTM models to learn spatial and temporal gesture patterns.
2. **YOLO-Enhanced Gesture Recognition:** Integrates YOLOv8-based hand detection with ROI extraction, CNN feature extraction and BiLSTM classification to improve spatial localization and reduce background interference.

The project focuses on comparing these approaches and understanding how dataset balancing, feature engineering and object detection affect gesture recognition performance.

## Key Features

- Recognition of nine predefined campus navigation gestures.
- Hand landmark extraction using MediaPipe.
- Spatial and temporal gesture modelling using CNN and BiLSTM architectures.
- Landmark-level data augmentation to address class imbalance.
- Joint angle and motion feature experimentation.
- YOLOv8-based hand detection and Region of Interest (ROI) extraction.
- Hybrid detection and classification pipeline.
- Webcam-based gesture prediction with temporal buffering and prediction stabilization.
- Mapping of recognized gestures to predefined campus navigation outputs.

## Methodology

### Phase 1: Landmark-Based Gesture Recognition

The first phase focuses on extracting hand landmarks from video frames using MediaPipe and using these features for sequence-based gesture classification.

The experiments include:

- Baseline BiLSTM model.
- BiLSTM with early stopping.
- BiLSTM with targeted landmark augmentation.
- CNN–BiLSTM architecture.
- CNN–BiLSTM with joint angle features.
- CNN–BiLSTM with angle and velocity features.
- Deeper CNN–BiLSTM architecture.

These experiments evaluate the impact of feature engineering, class imbalance and model complexity on gesture recognition performance.

### Phase 2: YOLO-Enhanced Gesture Recognition

The second phase introduces YOLOv8-based hand detection to improve the spatial consistency of the input data.

The pipeline consists of:

1. Video frame extraction.
2. Hand detection using YOLOv8.
3. Hand Region of Interest (ROI) extraction.
4. Feature extraction using a CNN-based approach.
5. Temporal sequence modelling using BiLSTM.
6. Gesture classification and navigation output mapping.

This approach combines object detection and temporal classification to reduce background interference and improve gesture recognition.

## Experimental Results

### Landmark-Based Models

| Model | Test Accuracy |
|---|---:|
| Baseline BiLSTM | 32.6% |
| BiLSTM with Early Stopping | 52.2% |
| BiLSTM with Targeted Augmentation | 67.09% |
| CNN–BiLSTM | 65.82% |
| CNN–BiLSTM with Angle Features | ~67% |
| CNN–BiLSTM with Angle and Velocity Features | 61% |
| Deeper CNN–BiLSTM | ~20% |

The experiments showed that targeted augmentation significantly improved performance. Joint angle features also improved geometric representation and classification stability, while adding velocity features did not improve performance on the limited dataset.

### YOLO-Enhanced Pipeline

| Metric | Result |
|---|---:|
| Test Accuracy | 100% |
| Precision | 1.00 |
| Recall | 1.00 |
| F1-score | 1.00 |
| YOLOv8 mAP@0.5 | ~0.97 |

**Important:** The hybrid pipeline's test set contained only six videos, approximately one per gesture class. Therefore, the reported 100% test accuracy represents proof-of-concept performance and should not be interpreted as evidence of generalization to unseen users or environments.

## Technology Stack

- **Programming Language:** Python
- **Computer Vision:** OpenCV, MediaPipe
- **Object Detection:** YOLOv8
- **Deep Learning:** TensorFlow / Keras
- **Sequence Modelling:** BiLSTM
- **Data Processing:** NumPy, Pandas
- **Model Evaluation:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn

## Repository Structure

```text
## Project Structure

Sign-Language-Campus-Navigation/
│
├── app/                # Application and real-time inference
├── models/             # YOLO, landmark-based and hybrid models
├── notebooks/          # Model development and experiments
├── report/             # Final project report
├── screenshots/        # Application screenshots (if included)
├── src/                # Landmark processing and model pipeline
│
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── .gitignore          # Files excluded from version control
```

The notebooks contain the implementation and experimentation workflows. The final report provides the detailed system design, dataset preparation, model development, evaluation and discussion.

## Dataset

The project dataset was collected specifically for the campus navigation task.

- 30 video recordings.
- Nine predefined campus gesture classes.
- Videos recorded in a controlled indoor environment.
- One signer.
- Fixed-length sequences of 30 frames for model training and evaluation.

The dataset was designed as a closed-set multi-class gesture recognition problem rather than a general-purpose sign language translation task.

The original video dataset is not included in this repository.

## Limitations

- The dataset contains recordings from a single signer.
- The dataset is relatively small and class distribution affects model performance.
- Data collection was conducted under controlled indoor conditions.
- The YOLO-enhanced pipeline was evaluated on a limited test set.
- Performance under varying lighting, backgrounds, occlusions and different signers requires further evaluation.
- YOLO-based detection introduces additional computational overhead.

The project should therefore be considered a proof-of-concept system rather than a fully generalized real-world sign language translation solution.

## Future Improvements

- Expand the dataset with multiple signers and diverse gesture styles.
- Evaluate performance under different lighting and background conditions.
- Increase the number of test samples for more reliable evaluation.
- Optimize detection and classification for lower-latency deployment.
- Extend the system to support a broader range of campus navigation gestures.

## Final Project Report

The final report provides a detailed explanation of the project, including:

- Literature review and problem formulation.
- System architecture and workflow.
- Dataset collection and preprocessing.
- Landmark-based model development.
- YOLOv8-based detection and ROI extraction.
- Experimental comparisons and evaluation metrics.
- Implementation, testing, discussion and conclusions.

[View the Final Project Report](report/Final Report.pdf)

## Academic Project

**Programme:** M.Sc. Data Science  
**Institution:** CHRIST (Deemed to be University), Bengaluru  
**Department:** Department of Statistics and Data Science  
**Project Guide:** Dr. Dalvin Vinoth Kumar A  
**Year:** 2025–2026
