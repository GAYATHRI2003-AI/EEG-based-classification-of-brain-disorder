# CogNet: EEG Analysis for Cognitive Disorder Classification

## Overview
CogNet is a deep learning-based framework designed to classify EEG signals into three cognitive states: **Normal, Mild Cognitive Impairment (MCI), and Dementia**. This project leverages deep learning methodologies such as **Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and Transformer models** to enhance the accuracy of cognitive state classification, enabling early detection of neurodegenerative diseases.

## Features
- **Automated EEG Classification**: Uses deep learning models to classify EEG signals into Normal, MCI, or Dementia.
- **Hybrid Model Approach**: Combines CNNs (spatial features), RNNs (temporal dependencies), and Transformer-based networks.
- **Feature Extraction & Preprocessing**: Utilizes **Discrete Wavelet Transform (DWT)**, bandpass filtering (1-40 Hz), and **Z-score normalization**.
- **Performance Evaluation**: Analyzes accuracy, precision, recall, F1-score, confusion matrix, and ROC curves.
- **Real-Time Cognitive Monitoring**: Can be integrated into clinical decision support systems or wearable EEG devices.

## Dataset
The system is trained on EEG data from the **Chung-Ang University Hospital EEG (CAUEEG) repository**, which contains **1,388 EEG recordings** with labeled cognitive states.
Later the dataset is manually classified based on the disorders.
The dataset is available at Kaggle Dataset - (https://www.kaggle.com/datasets/gayathrideviboopathy/signals-dataset).
Download and unzip the dataset before using it.

## Methodology
1. **Data Collection & Preprocessing**:
   - EEG signals are filtered (1-40 Hz), artifacts removed using **Independent Component Analysis (ICA)**.
   - Feature extraction through **Wavelet Transform** and **Fourier Transform**.
2. **Deep Learning Models**:
   - **CNN Model**: Extracts spatial features from EEG data.
   - **RNN (LSTM & GRU) Model**: Captures temporal dependencies.
   - **Transformer-Enhanced RNN Model**: Uses attention mechanisms for improved classification.
3. **Model Training & Evaluation**:
   - Training with **Adam optimizer**, **categorical cross-entropy loss**, **batch size: 16**, **epochs: 20**.
   - Performance assessed using standard evaluation metrics.

## Results
- **CNN Model**: **82.17% Accuracy** (Best performer)
- **RNN Model (LSTM/GRU)**: **81.27% Accuracy**
- **RNN + Transformer**: **79.04% Accuracy**
- The **CNN model** demonstrated superior performance in extracting spatial EEG features, while the **RNN and Transformer-based models** captured sequential dependencies.

## Installation & Usage
### Prerequisites
- Python 3.x
- TensorFlow, PyTorch, Keras
- NumPy, SciPy, Pandas, Matplotlib
- MNE-Python (for EEG signal processing)
- Flask/Django (for deployment)

### Running the Model
The training process involves loading EEG data, preprocessing it, extracting features, and training the deep learning models. 
The trained models can then be used to classify EEG signals into Normal, MCI, or Dementia states.


## Applications
- **Early Diagnosis of Neurodegenerative Disorders**
- **Real-Time Cognitive Monitoring in Healthcare**
- **Non-Invasive, Cost-Effective Brain Analysis**
- **Integration with Wearable EEG Devices**

## Note
Add the dataset in your Google Drive, unzip it , copy the path and use it in the code.

## Acknowledgments
Dr. Nagaraj Halemani and Chung-Ang University Hospital EEG (CAUEEG) repository for providing the comprehensive EEG dataset

## License
This project is licensed under the **MIT License**. Feel free to contribute and improve!

## Contact
For any queries or contributions, please reach out to **gaya3devi.2003b@gmail.com**.

---


This README provides a structured overview of the CogNet project, covering key features, methodology, dataset, installation steps, results, and contributors. Let me know if you need any modifications!

