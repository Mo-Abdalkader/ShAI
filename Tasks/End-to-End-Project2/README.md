# Documentation

## Content Table
1. Problem Statement
2. Data Description
3. Methodology
   1. Data Preprocessing
   2. Data Visualization
   3. Model Selection & Training
   4. Model Evaluation
   5. Model Fine-Tune
   6. Model predictions
   7. Submission Dataset (For Competition)
4. Summary

### 1. Problem Statement
In the era of digital music streaming, categorizing music into genres is crucial for organizing large music libraries, improving recommendation systems, and enhancing user experience. However, manual classification is time-consuming and error-prone, given the vast number of tracks produced daily.

This project aims to solve the problem of automatically classifying music tracks into genres using their acoustic and perceptual features. By applying machine learning techniques, we aim to build a model that can predict the genre of a song based on characteristics such as danceability, energy, tempo, and others.

The challenge lies in handling a dataset with a variety of features, ensuring proper preprocessing, and selecting the best model that provides a high level of accuracy while maintaining a balanced trade-off between precision and recall.
***

### 2. Data Description
The dataset used in this project contains various musical features that describe different aspects of music tracks. These features provide both acoustic and perceptual insights into the nature of each track, enabling us to classify them into genres. Below is a description of the key features:


| **Feature**            | **Description**                                                                                       |
|------------------------|-------------------------------------------------------------------------------------------------------|
| `Id`                   | Unique identifier for each track.                                                                     |
| `Artist Name`          | The name of the artist who performed the track.                                                        |
| `Track Name`           | The title of the track.                                                                                |
| `Popularity`           | A numerical value indicating the track’s popularity (higher values indicate more popularity).          |
| `danceability`         | How suitable the track is for dancing, based on tempo, rhythm, and beat consistency.                   |
| `energy`               | Measures the intensity and activity of the track (values range from 0 to 1).                           |
| `key`                  | The musical key in which the track is composed, represented as a numerical value.                      |
| `loudness`             | Overall volume of the track, measured in decibels (dB).                                                |
| `mode`                 | Indicates whether the track is in a major (1) or minor (0) scale.                                      |
| `speechiness`          | Measures the presence of spoken words in the track (higher values indicate more speech).               |
| `acousticness`         | Confidence that the track is acoustic (values close to 1 indicate a higher probability of being acoustic).|
| `instrumentalness`     | Predicts the likelihood that the track is instrumental (no vocals).                                    |
| `liveness`             | Detects the presence of an audience or background noise, indicating live performance.                  |
| `valence`              | Describes the musical positiveness or emotional tone of the track (higher values suggest happier tones).|
| `tempo`                | The speed or pace of the track, measured in beats per minute (BPM).                                    |
| `duration_in min/ms`   | Duration of the track, in either minutes or milliseconds.                                              |
| `time_signature`       | The number of beats per measure, indicating the track’s time signature.                                |
| `Class`                | The target variable representing the genre of the track (classification label).                        |

***
### 3. Methodology
#### 3.1 Data Preprocessing
> [!NOTE]  
> Data Preprocessing is the process of cleaning, transforming, and organizing raw data to prepare it for analysis or machine learning. This step ensures that the data is accurate, consistent, and in a format suitable for processing, which helps improve the quality and performance of analytical models.


##### 3.1.1 Display sample
##### 3.1.2 Display Info
##### 3.1.3 Display Description
##### 3.1.4 Check Nulls
##### 3.1.5 Check Duplicates
##### 3.1.6 Drop Useless Columns

#### 3.2 Data Visualization
> [!NOTE]  
> Data Visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.

##### 3.2.1 Histogram
##### 3.2.2 Correlation Heatmap
##### 3.2.3 Display Nulls
##### 3.2.4 Check Outliers
##### 3.2.4.1 Using IQR
##### 3.2.4.2 Using Scatter Plot
##### 3.2.4.3 Using Boxplot
##### 3.2.5 Plot Features by Class (Count and Average)
##### 3.2.6 Splitting
##### 3.2.7 Handle Unbalanced Data
##### 3.2.8 Scaling
##### 3.2.9 Features Selection

#### 3.3 Model Selection & Training
> [!NOTE]  
> Model selection is the process of choosing the best algorithm for a task based on performance metrics. 
Training is the process of fitting the model to data to learn patterns and make predictions.

##### 3.3.1 Random Forest
##### 3.3.2 XGBoost
##### 3.3.3 Gradient Boosting
##### 3.3.4 KNN
##### 3.3.5 CatBoost
##### 3.3.6 LGBM

#### 3.4 Evaluation
> [!NOTE]  
> Evaluation is the process of assessing a model's performance by comparing its predictions to actual outcomes using metrics like accuracy, precision, recall, or F1 score. It helps determine how well the model generalizes to new, unseen data.

#### 3.5 Fine-Tuning
> [!NOTE]  
> Fine-tuning is the process of optimizing a model by adjusting its hyperparameters to improve performance and achieve better accuracy, precision, or other evaluation metrics. It typically follows initial model training and evaluation.

#### 3.6 Model predictions
#### 3.7 Submission Dataset (For Competition)

