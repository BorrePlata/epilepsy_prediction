# Epileptic Seizure Prediction with Machine Learning

## Project Overview

The primary goal of this project is to develop a system that uses electroencephalogram (EEG) data to predict epileptic seizures at least five seconds in advance. This system should also be able to identify different types of epilepsy and demonstrate its functionality through real-time simulation using the most accurate data available.

## Project Description

### 1. Data Acquisition

We start the project by acquiring high-quality EEG data from open and reliable sources. We use APIs and direct links to EEG datasets such as the Temple University Hospital EEG Corpus, the CHB-MIT Scalp EEG Database, and the Epilepsy Ecosystem (EpiEpi).

### 2. Data Preprocessing

Once the data is obtained, the next step is preprocessing. This includes cleaning the data, removing noise, normalizing, and segmenting it into time windows. We use techniques like filtering and scaling to prepare the data for modeling.

### 3. Model Construction and Training

To predict epileptic seizures, we build a machine learning model using a combination of convolutional neural networks (CNN) and LSTM to capture both spatial and temporal features of the EEG data. The model is trained with preprocessed data.

### 4. Real-Time Simulation

To demonstrate the system's functionality, we simulate real-time data and use the trained model to make predictions. This simulation shows how the system detects an imminent seizure and adjusts the necessary parameters.

### 5. Iterative Adjustments

To improve model performance, we implement an iterative adjustment approach. We use hyperparameter search techniques and cross-validation to optimize the model with each iteration.

## Significant Findings

### Correlation Heatmap between EEG Channels

This heatmap shows the correlations between different features extracted from the EEG signals of various channels. The lighter areas indicate a strong positive correlation, while the darker areas represent a negative correlation. The strong correlation between certain channels may suggest that they are measuring similar brain activities, which is relevant for identifying patterns associated with epileptic episodes.

![Correlation Heatmap](./neuroscience/figures/correlation_heatmap.png)

### EEG Feature Distribution

These graphs show the distribution of various features extracted from EEG signals, such as mean, standard deviation, maximum value, minimum value, and energy. Most distributions are left-skewed, suggesting that most values are low, with some extremely high outliers. These outliers could be associated with epileptic events or artifacts in the data.

![EEG Features](./neuroscience/figures/eeg_features.png)

### EEG Signals

The graphs show the amplitude of EEG signals over time for five different channels. The signals present peaks at certain time intervals, which could correspond to epileptic events. Analyzing these signals in the time domain is essential for identifying and characterizing these events.

![EEG Signals](./neuroscience/figures/eeg_signals.png)

### Frequency Analysis of EEG Signals

This graph shows the power spectral density (PSD) of EEG signals for different channels, indicating how the signal's energy is distributed across different frequencies. The different channels show similar frequency patterns, but with some variations. These differences can help identify specific brain activity patterns associated with epilepsy.

![Frequency Analysis](./neuroscience/figures/frequency_analysis.png)

### 3D Correlation Heatmap

This 3D heatmap representation helps visualize the correlation between EEG channels more clearly, showing the intricate relationships between different features. This visualization can be rotated and examined from various angles for better analysis.

![3D Correlation Heatmap](./neuroscience/figures/correlation_heatmap_3d.png)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Author: Samuel Plata  
Email: [Your Email]  
LinkedIn: [Your LinkedIn]

We aim to develop an effective and precise tool for early detection of epileptic seizures, which not only predicts events in advance but also identifies different types of epilepsy. We seek to improve the management and treatment of epilepsy, enabling healthcare professionals to better anticipate and respond to epileptic episodes.
