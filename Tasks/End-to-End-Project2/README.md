# Project Documentation: Music Genre Classification

## Table of Contents
1. Problem Statement
2. Data Description
3. Methodology  
   1. Data Preprocessing  
   2. Data Visualization  
   3. Model Selection & Training  
   4. Model Evaluation  
   5. Model Fine-Tuning  
   6. Model Predictions  
   7. Submission Dataset (For Competition)  
4. Summary

### 1. Problem Statement
> [!NOTE]  
In the era of digital music streaming, categorizing music into genres is crucial for organizing large music libraries, improving recommendation systems, and enhancing user experience. However, manual classification is time-consuming and error-prone, given the vast number of tracks produced daily.

This project aims to solve the problem of automatically classifying music tracks into genres using their acoustic and perceptual features. By applying machine learning techniques, we aim to build a model that can predict the genre of a song based on characteristics such as danceability, energy, tempo, and others.

The challenge lies in handling a dataset with a variety of features, ensuring proper preprocessing, and selecting the best model that provides a high level of accuracy while maintaining a balanced trade-off between precision and recall.

---

### 2. Data Description
> [!NOTE]  
The dataset used in this project contains various musical features that describe different aspects of music tracks. These features provide both acoustic and perceptual insights into the nature of each track, enabling us to classify them into genres. Below is a description of the key features:

| **Feature**            | **Description**                                                                                       |
|------------------------|-------------------------------------------------------------------------------------------------------|
| `Id`                   | Unique identifier for each track.                                                                     |
| `Artist Name`          | The name of the artist who performed the track.                                                        |
| `Track Name`           | The title of the track.                                                                                |
| `Popularity`           | A numerical value indicating the track’s popularity (higher values indicate more popularity).          |
| `Danceability`         | How suitable the track is for dancing, based on tempo, rhythm, and beat consistency.                   |
| `Energy`               | Measures the intensity and activity of the track (values range from 0 to 1).                           |
| `Key`                  | The musical key in which the track is composed, represented as a numerical value.                      |
| `Loudness`             | Overall volume of the track, measured in decibels (dB).                                                |
| `Mode`                 | Indicates whether the track is in a major (1) or minor (0) scale.                                      |
| `Speechiness`          | Measures the presence of spoken words in the track (higher values indicate more speech).               |
| `Acousticness`         | Confidence that the track is acoustic (values close to 1 indicate a higher probability of being acoustic).|
| `Instrumentalness`     | Predicts the likelihood that the track is instrumental (no vocals).                                    |
| `Liveness`             | Detects the presence of an audience or background noise, indicating live performance.                  |
| `Valence`              | Describes the musical positiveness or emotional tone of the track (higher values suggest happier tones).|
| `Tempo`                | The speed or pace of the track, measured in beats per minute (BPM).                                    |
| `Duration_in_min/ms`   | Duration of the track, in either minutes or milliseconds.                                              |
| `Time_Signature`       | The number of beats per measure, indicating the track’s time signature.                                |
| `Class`                | The target variable representing the genre of the track (classification label).                        |

---

### 3. Methodology

#### 3.1 Data Preprocessing
> [!NOTE]  
Data preprocessing ensures that raw data is cleaned, transformed, and organized to make it suitable for machine learning. This process involves handling missing values, normalizing features, and preparing data for further analysis.

- **Displaying Sample Data:** A glimpse of the dataset is presented to understand its structure.
- **Displaying Info:** The structure and types of the data are checked using the `info()` method to identify missing or misformatted data.
- **Checking Null Values:** Identifying if any features contain missing values.
- **Checking for Duplicates:** Ensuring that no duplicate records exist in the dataset.
- **Dropping Useless Columns:** Removing any columns that do not contribute to the classification task.
  
#### 3.2 Data Visualization
> [!NOTE]  
Data visualization helps reveal hidden patterns and relationships between features.

- **Histogram:** Visualize the distribution of different features.
- **Correlation Heatmap:** Understand how features are correlated with one another.
- **Checking for Outliers:** Detecting anomalies in data using methods such as IQR (Interquartile Range), scatter plots, and box plots.
- **Plot Features by Class:** Examining class-wise statistics for each feature.
- **Handling Imbalanced Data:** Techniques like oversampling or undersampling to balance class distribution.
- **Scaling:** Normalizing or standardizing features before feeding them into machine learning models.
- **Feature Selection:** Selecting the most relevant features using statistical techniques.

#### 3.3 Model Selection & Training
> [!NOTE]  
Model selection is critical to finding the best algorithm for the task at hand. We explored various classifiers:

- **Random Forest**
- **XGBoost**
- **Gradient Boosting**
- **K-Nearest Neighbors (KNN)**
- **CatBoost**
- **LGBM (LightGBM)**

Each model was trained using a combination of hyperparameter tuning methods such as Grid Search and Randomized Search to optimize performance.

#### 3.4 Model Evaluation
> [!NOTE]  
Model evaluation is performed using standard metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

These metrics provide insights into how well the model generalizes to unseen data.

#### 3.5 Fine-Tuning
> [!NOTE]  
Fine-tuning involves further optimizing model hyperparameters to maximize performance. Techniques such as cross-validation were used to select the best-performing models.

#### 3.6 Model Predictions
> [!NOTE]  
The trained model is used to make predictions on unseen data (the test set) to evaluate its effectiveness in real-world scenarios.

#### 3.7 Submission Dataset (For Competition)
> [!NOTE]  
For the competition, the final predictions are stored in a submission dataset that includes the track IDs and their predicted genres.

---

### 4. Summary
> [!NOTE]  
This project demonstrated how machine learning techniques can be applied to the problem of genre classification using acoustic and perceptual features of music tracks. The use of data preprocessing, feature selection, and model fine-tuning contributed to building a highly accurate classification system that can predict the genre of a song based on its characteristics.
