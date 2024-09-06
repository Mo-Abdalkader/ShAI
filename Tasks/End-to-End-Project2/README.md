# Music Genre Classification Project

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Project Workflow](#project-workflow)
    1. [Data Preprocessing](#data-preprocessing)
    2. [Model Selection](#model-selection)
    3. [Model Evaluation](#model-evaluation)
    4. [Cross-Validation and Hyperparameter Tuning](#cross-validation-and-hyperparameter-tuning)
4. [Results](#results)
5. [Conclusion](#conclusion)
6. [Future Work](#future-work)
7. [Installation and Setup](#installation-and-setup)
8. [Author](#author)

---

## Project Overview

This project focuses on the classification of music tracks into specific genres using various machine learning techniques. The dataset consists of musical features such as danceability, energy, loudness, and other acoustic and perceptual characteristics. The goal is to use these features to build a predictive model that accurately classifies a track into its respective genre.

Music classification can have numerous applications in the music industry, such as recommendation systems, organizing libraries, or creating genre-based playlists.

## Dataset Description

The dataset used for this project contains multiple features that describe different aspects of a music track. Below is a detailed description of the features:

- **artist**: Name of the artist who performed the track.
- **song**: Title of the song.
- **popularity**: A numerical value indicating the song's popularity, where higher values suggest more popularity.
- **danceability**: Describes how suitable a track is for dancing, considering factors like rhythm and tempo.
- **energy**: A value between 0 and 1 indicating the intensity and activity of the track.
- **key**: The musical key in which the track is composed (encoded as integers).
- **loudness**: Overall loudness of the track in decibels.
- **mode**: Indicates whether the track is in a major (1) or minor (0) key.
- **speechiness**: Detects the presence of spoken words in a track.
- **acousticness**: Confidence measure of how acoustic the track is.
- **instrumentalness**: Predicts the likelihood of the track being instrumental.
- **liveness**: Detects the presence of an audience in the track.
- **valence**: Measures the musical positiveness of the track.
- **tempo**: The speed or pace of the track measured in beats per minute (BPM).
- **time_signature**: The number of beats in each measure of the track.
- **Class**: The target variable representing the genre of the track.

The dataset was divided into training and testing sets, used for model training and evaluation.

## Project Workflow

The project follows a systematic workflow to transform raw data into a predictive model. Below is a breakdown of the major steps involved:

### 1. Data Preprocessing

Data preprocessing is a critical step in ensuring that the dataset is clean and well-formatted for machine learning models. In this project, the following preprocessing steps were taken:

- **Handling Missing Data**: Missing values were handled using various imputation methods, such as `SimpleImputer` for mean/mode filling and `KNNImputer` for more sophisticated imputations.
- **Scaling and Normalization**: Continuous features were scaled using `StandardScaler` and `MinMaxScaler` to ensure uniformity across different ranges of values.
- **Categorical Encoding**: Label encoding and one-hot encoding were used to convert categorical variables (such as genre) into numeric format.
- **Dimensionality Reduction**: Principal Component Analysis (PCA) was employed to reduce the dimensionality of the dataset, ensuring that only the most important features were used in model training.

### 2. Model Selection

Various machine learning algorithms were explored to identify the best model for this task. The following models were tested:

- **Random Forest Classifier**: A robust ensemble learning method that builds multiple decision trees and combines their outputs for better performance.
- **Gradient Boosting Classifier**: A boosting technique that builds models sequentially, each attempting to correct the errors of its predecessor.
- **LightGBM and XGBoost**: Highly efficient gradient-boosting algorithms that are optimized for both speed and performance.
- **Support Vector Classifier (SVC)**: A classifier that attempts to find the optimal hyperplane that best separates the classes in the data.
- **K-Nearest Neighbors (KNN)**: A simple, instance-based learning algorithm that classifies based on the majority class among the K-nearest neighbors.
  
Each model was evaluated based on accuracy, precision, recall, and F1 score.

### 3. Model Evaluation

Model performance was evaluated using a variety of metrics:

- **Accuracy**: The proportion of correct predictions made by the model.
- **Precision**: The number of true positives divided by the sum of true and false positives.
- **Recall**: The number of true positives divided by the sum of true positives and false negatives.
- **F1-Score**: The harmonic mean of precision and recall.
- **Confusion Matrix**: A summary of the model’s predictions compared to the actual labels, used to calculate false positives, false negatives, etc.

Additionally, **ConfusionMatrixDisplay** was used to visualize model performance.

### 4. Cross-Validation and Hyperparameter Tuning

To ensure the model's robustness, cross-validation was applied using techniques like Stratified K-Folds. This divides the dataset into multiple subsets to ensure that the model performs consistently.

- **GridSearchCV** and **RandomizedSearchCV** were used for hyperparameter tuning, which helped find the best model parameters and improve the model’s overall performance.

## Results

After evaluating all models, the final selected model achieved the following performance metrics on the test dataset:

- **Test Accuracy**: 87.73%
- **Precision**: 89.94%
- **Recall**: 87.73%
- **F1 Score**: 87.73%

The model demonstrated strong predictive capability with a good balance between precision and recall.

## Conclusion

This project successfully applied machine learning techniques to classify music tracks into genres. The final model achieved an accuracy of 87.73%, with further potential for improvement through deep learning or more sophisticated feature engineering.

## Future Work

Several avenues for future work have been identified:

- **Deep Learning Models**: Implementing neural networks or Convolutional Neural Networks (CNNs) for further performance improvement.
- **Feature Engineering**: Creating new features or exploring advanced feature extraction methods from audio data.
- **Handling Class Imbalance**: Further techniques such as Synthetic Minority Over-sampling Technique (SMOTE) or more advanced resampling methods could be applied to improve model performance in cases where some genres are under-represented.

## Installation and Setup

To run this project, you will need the following libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost
```

To run the notebook, clone the repository and make sure the dataset is in the music-data/ folder:
```bash
git clone https://github.com/YourUsername/Music-Genre-Classification.git
cd Music-Genre-Classification
```
Then, open and run the notebook using Jupyter Notebook.

