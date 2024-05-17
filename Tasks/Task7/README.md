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
The objective of this project is to develop a predictive model that accurately estimates the price of diamonds based on their attributes. Given the significant variability in diamond prices due to differences in physical characteristics and quality, the goal is to utilize machine learning techniques to identify patterns and relationships within the dataset that can predict the price of a diamond. This will aid consumers and retailers in making informed decisions regarding diamond purchases and sales.

***

### 2. Data Description
The dataset contains detailed information on nearly 54,000 diamonds, including their prices and various physical and quality attributes. Each record in the dataset represents a unique diamond and includes the following features:
```markdown
price -----------------------------------------------------> $326 : $18,823

carat weight ---------------------------------------------->  0.2 : 5.01
cut quality -----------------------------------------------> (Fair, Good, Very Good, Premium, Ideal)
color 'diamond colour' ------------------------------------> (J (worst), I, H, G, F, E, D (best))
clarity "how clear the diamond is" ------------------------> (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))

x 'length' ------------------------------------------------> 0 : 10.74 mm
y 'width'  ------------------------------------------------> 0 : 58.90 mm
z 'depth'  ------------------------------------------------> 0 : 31.80 mm

depth 'total depth percentage' ----------------------------> 43 : 79 (%)
       = z / mean(x, y) = 2 * z / (x + y)
                             
table 'width of top of diamond relative to widest point' --> 43 : 95
```

***

### 3. Methodology
#### 3.1 Data Preprocessing
Data preprocessing is a crucial step in preparing a dataset for machine learning. It involves cleaning, transforming, and organizing the data to ensure that it is in the best possible condition for model training.

##### 3.1.1 View the Data, Nulls Check, Duplicates Check
**Code :**
```python
# DataFrame Summary
train_df.info()

# Summary Statistics for Numerical Columns
train_df.describe()

# Summary Statistics for Categorical Columns
train_df.describe(include='O')

# Missing Values Count
train_df.isna().sum()

# Duplicate Rows Count
train_df.duplicated().sum()

# Random Sample of DataFrame
train_df.sample(5)
```
##### 3.1.2 Zeros Handling
```python
# Count the number of rows in the DataFrame where any of the dimensions (length, width, or depth) of a diamond is equal to zero.
print("Total number of (X & Y & Z) Zeros in train_df = ", len(train_df[(train_df['z'] == 0) | (train_df['y'] == 0) | (train_df['x'] == 0)]))
print("Total number of (X & Y & Z) Zeros in test_df  = ", len(test_df[(test_df['z'] == 0) | (test_df['y'] == 0) | (test_df['x'] == 0)]))
```
```markdown
Total number of (X & Y & Z) Zeros in train_df = 17
Total number of (X & Y & Z) Zeros in test_df  = 3
```
To handle these values we use this function, It effectively replaces zero values in the dimension column of the test dataset with the mean value of that dimension for diamonds with similar carat weights in the training dataset.

```python
def handle_test_xyz(col, df="test_df"):
    sub_test_df = df[df[col] == 0]
    for ID in sub_test_df["Id"]:
        carat_ = float(sub_test_df[sub_test_df["Id"] == ID]["carat"])
        
        carat_min = carat_ - 0.1
        carat_max = carat_ + 0.1
        
        sub_train_df = train_df[(train_df["carat"] >= carat_min) & (train_df["carat"] <= carat_max) ]
        
        df.loc[df["Id"] == ID, col] = sub_train_df[col].mean()
        sub_test_df.loc[sub_test_df["Id"] == ID, col] = sub_train_df[col].mean()

handle_test_xyz("x", train_df)
handle_test_xyz("y", train_df)
handle_test_xyz("z", train_df)

handle_test_xyz("x", test_df)
handle_test_xyz("y", test_df)
handle_test_xyz("z", test_df)
```
##### 3.1.3 Drop Meaningless Columns
```python
# Dropping the "Id" column, It typically serves as an identifier and does not contain any meaningful information  
train_df = train_df.drop(["Id"], axis=1)

# Save test_Df IDs for the Competition submission
test_IDs = test_df["Id"]
test_df  = test_df.drop(["Id"], axis=1)
```
##### 3.1.4 Make a DataFrame check point
```python
train_df_copy = train_df.copy()
```
##### 3.1.5 Label Encoding
```python
# Display The Categories of Each Categorical Column
print(train_df["cut"].unique())
print(train_df["color"].unique())
print(train_df["clarity"].unique())
```
```markdown
['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
["J", "I", "H", "G", "F", "E", "D"]
["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
```
```python
# Save These Categories to be Used in Visualization
cut_column_unique_ordered = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_column_unique_ordered = ["J", "I", "H", "G", "F", "E", "D"]
clarity_column_unique_ordered = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]


# Encoding Each Column
train_df["cut"].replace({"Fair":0, "Good":1, "Very Good":2, "Premium":3, "Ideal":4}, inplace=True)
test_df ["cut"].replace({"Fair":0, "Good":1, "Very Good":2, "Premium":3, "Ideal":4}, inplace=True)

train_df["color"].replace({"J":0, "I":1, "H":2, "G":3, "F":4, "E":5, "D":6}, inplace=True)
test_df ["color"].replace({"J":0, "I":1, "H":2, "G":3, "F":4, "E":5, "D":6}, inplace=True)

train_df["clarity"].replace({"I1":0, "SI2":1, "SI1":2, "VS2":3, "VS1":4, "VVS2":5, "VVS1":6, "IF":7}, inplace=True)
test_df ["clarity"].replace({"I1":0, "SI2":1, "SI1":2, "VS2":3, "VS1":4, "VVS2":5, "VVS1":6, "IF":7}, inplace=True)
```

#### 3.2 Data Visualization
##### 3.2.1 Correlation Heatmap
Visulizes the correlation between the numerical features 
``` python
plt.subplots(figsize=(12, 8))
sns.heatmap(train_df.corr(), annot=True)
```
**Insights :**
- There is a strong positive correlation between the price (target column) and the numerical features carat, x, y, and z
- There is an inverse relationship between the categorical features

##### 3.2.2 Histogram
Visulizes the distribution of data
``` python
train_df.loc[:, ["carat", "depth", "table", "x", "y", "z", "price"]].hist(figsize=(20, 15), bins=50)
```

**Insights :**
- The presence of right skewness in most numerical features suggests a potential presence of outliers
- The varying ranges of the data indicate a need for scaling to ensure that all features contribute equally to the model

##### 3.2.3 BoxPlot
Visulizes the distribution of data based on a five-number summary: minimum, first quartile (Q1), median (Q2), third quartile (Q3), and maximum. 
It is useful for identifying outliers and understanding the spread and skewness of the data.

``` python
fig, axes = plt.subplots(2, 3, figsize=(16, 14))
for i, cols in enumerate(["carat","depth", "table", "x", "y", "z"], 1):
    sns.boxplot(data=train_df, x=cols)
    plt.subplot(2, 3, i)

plt.show()
```
**Insights :**
- There are alot of outliers in the data espacially in carat, depth and price

##### 3.2.4 Interquartile range (IQR)
Can be used to detect the outliers
``` python
def get_outliers(col, train_df = train_df):
    Q1 = train_df[col].quantile(0.25)
    Q3 = train_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    iqr_outliers = train_df[(train_df[col] < lower_bound) | (train_df[col] > upper_bound)]
    
    return lower_bound, upper_bound,len(iqr_outliers)

for col in train_df.columns[:7]:
    lower_bound, upper_bound, outliers = get_outliers(col)
    print(f"Number of outliers detected based on IQR method ({str(col).center(8)}) : ", str(outliers).rjust(4), f"Lower = {str(round(lower_bound, 3)).rjust(10)}", f"Upper = {str(round(upper_bound, 2)).rjust(10)}")        
```
This code is designed to:
1) Detect Outliers: Identify potential outliers in the first seven columns of the DataFrame using the IQR method.
2) Summarize Outliers: Print a summary of the outlier detection results for each of these columns, including the count of outliers and the calculated lower and upper bounds for identifying outliers.

**Insights :**
- As we thought before there are alot of outliers
- in price column there are  2844 outliers
- in depth column there are  2182 outliers
- in carat column there are  1504 outliers
  
##### 3.2.5 Outliers Handling
Using Scatter plot we can determine the boundaries of outliers

``` python
sns.scatterplot(x=train_df["carat"], y=train_df["price"])        

print(f"Carat Outliers : {len(train_df[train_df['carat'] > 3])}") # Carat Outliers : 21
train_df = train_df[train_df["carat"] <= 3 ]
```
- Outliers were observed in the carat feature with values greater than 3.
- A total of 21 rows containing such outliers were dropped from the dataset.

``` python
sns.scatterplot(x=train_df["depth"], y=train_df["price"])

print(f"Depth Outliers : {len(train_df[(train_df['depth'] < 50) | (train_df['depth'] > 75)])}") # Depth Outliers : 14
train_df = train_df[(train_df["depth"] >= 50) & (train_df["depth"] <= 75)]    
```
- Outliers were noticed in the depth feature less than 50 and greater than 75.
- A total of 14 rows containing these outliers were removed.

``` python
sns.scatterplot(x=train_df["table"], y=train_df["price"])

print(f"Table Outliers : {len(train_df[(train_df['table'] < 50) | (train_df['table'] > 70)])}") # Table Outliers : 9
train_df = train_df[(train_df["table"] >= 50) & (train_df["table"] <= 70)]     
```
- Outliers were observed in the table feature less than 50 and greater than 70.
- A total of 9 rows containing these outliers were removed.

``` python
sns.scatterplot(data=train_df, x="x", y="price")

print(f"X Outliers : {len(train_df[train_df['x'] > 9])}") # X Outliers : 10
train_df = train_df[train_df["x"] <= 9]
```
- Outliers were identified in the x feature after the value 9.
- A total of 10 rows containing these outliers were removed.

``` python
sns.scatterplot(data=train_df, x="y", y="price")

print(f"Y Outliers : {len(train_df[train_df['y'] > 9])}") # Y Outliers : 0
train_df = train_df[train_df["y"] <= 9]
```
- Outliers were noticed in the y feature after the value 10.
- No rows were found to contain such outliers.

``` python
sns.scatterplot(data=train_df, x="z", y="price")

print(f"Z Outliers : {len(train_df[(train_df['z'] < 2) | (train_df['z'] > 7)])}") # Z Outliers : 0
train_df = train_df[(train_df["z"] >= 2) & (train_df["z"] <= 7)]
```
- Outliers were observed in the z feature less than 2 and greater than 7.
- No rows were found to contain such outliers.


##### 3.2.6 Countplot
``` python
``` python
mean_price_by_cut = train_df_copy.groupby("cut")["price"].mean().reindex(cut_column_unique_ordered)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.countplot(data=train_df_copy, x="cut", order=cut_column_unique_ordered, ax=axes[0], palette='rocket')
axes[0].set_title("Count of Diamonds by Cut")
axes[0].set_xlabel("Cut")
axes[0].set_ylabel("Count")
sns.barplot(x=mean_price_by_cut.index, y=mean_price_by_cut.values, ax=axes[1], palette='rocket')
axes[1].set_title("Mean Price of Diamonds by Cut")
axes[1].set_xlabel("Cut")
axes[1].set_ylabel("Mean Price")

plt.show()
```
This code will generate two side-by-side plots: one showing the count of diamonds for each cut category and another showing the mean price of diamonds for each cut category.
We found there are 0 rows So,we drop it to make sure that there is not

**Insights :**
- cut (J) that have the smallest count has the largest price average
We do so for color and clarity and found that the price dosn't depend on ctegorical features (as there are inverse relation between them and the price )

##### 3.2.7 Feature Engineering
``` python
#creating volume column in both tain and test dataframe and drop the x,y and z
train_df_copy["vol"] = train_df_copy["x"] * train_df_copy["y"] * train_df_copy["z"] # For Categorical Visualization
# train_df_copy.drop(["x", "y", "z"], axis=1, inplace=True)

train_df["vol"] = train_df["x"] * train_df["y"] * train_df["z"]                     # For Training
train_df.drop(["x", "y", "z"], axis=1, inplace=True)

test_df["vol"] = test_df["x"] * test_df["y"] * test_df["z"]                         # For Testing
test_df.drop(["x", "y", "z"], axis=1, inplace=True)
```
this reduces the dimensionality as we drop 3 columns and replaces it with one column

##### 3.2.8 Countplot 
``` python
This code will generate three side-by-side bar plots showing the mean volume of diamonds grouped by cut, color, and clarity, respectively.
mean_vol_by_cut     = train_df_copy.groupby("cut")["vol"].mean().reindex(cut_column_unique_ordered)
mean_vol_by_color   = train_df_copy.groupby("color")["vol"].mean().reindex(color_column_unique_ordered)
mean_vol_by_clarity = train_df_copy.groupby("clarity")["vol"].mean().reindex(clarity_column_unique_ordered)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
# Mean volume by cut
sns.barplot(x=mean_vol_by_cut.index, y=mean_vol_by_cut.values, palette='rocket', ax=axes[0])
axes[0].set_title("Mean Volume of Diamonds by Cut")
axes[0].set_xlabel("Cut")
axes[0].set_ylabel("Mean Volume")
# Mean volume by color
sns.barplot(x=mean_vol_by_color.index, y=mean_vol_by_color.values, palette='rocket', ax=axes[1])
axes[1].set_title("Mean Volume of Diamonds by Color")
axes[1].set_xlabel("Color")
axes[1].set_ylabel("Mean Volume")
# Mean volume by clarity
sns.barplot(x=mean_vol_by_clarity.index, y=mean_vol_by_clarity.values, palette='rocket', ax=axes[2])
axes[2].set_title("Mean Volume of Diamonds by Clarity")
axes[2].set_xlabel("Clarity")
axes[2].set_ylabel("Mean Volume")

plt.show()
```
**insights :**
- The category with highest volume has highest price

##### 3.2.9 Scaling & Spliting 
##### 3.2.9.1 Scaling
It helps algorithms converge faster and prevents features with larger scales from dominating those with smaller scales.

```python
y = train_df["price"]
X = train_df.drop(["price"], axis=1)


scaler = StandardScaler()
scaler.fit(X)

scaled_X = scaler.fit_transform(X)
submission_data = scaler.fit_transform(test_df)
```

##### 3.2.9.2 Spliting
Splits the dataset into training and testing set
- Training set : Used to train the machine learning model.
- Testing set : Used to evaluate the trained model's performance on unseen data.

```python
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, random_state=42, test_size = 0.2)
```
***

#### 3.3 Model Selection & Training
Choose a set of candidate models suitable for the task
##### 3.3.1 Linear Regression
Used to find the best-fitting straight line that describes the relationship between the variables

```python
from sklearn.linear_model import LinearRegression

linearReg_model = LinearRegression()
linearReg_model.fit(X_train, y_train)

linearReg_y_pred = linearReg_model.predict(X_test)
linearReg_RMSE = np.sqrt(mean_squared_error(y_test, linearReg_y_pred))
linearReg_MAE = mean_absolute_error(y_test, linearReg_y_pred)


print("Linear Regression RMSE :", linearReg_RMSE)
print("Linear Regression MAE  :", linearReg_MAE, "\n")

for index, (real_value, predicted_value) in zip(range(10), zip(y_test, linearReg_y_pred)):
    print(f"Real value is {str(round(real_value)).rjust(5)} | The predicted value is {str(round(predicted_value)).rjust(5)}")
```
**Output :**
```markdown
Linear Regression RMSE : 1223.9802099892302
Linear Regression MAE  : 859.7176167047971 

Real value is  3390 | The predicted value is  4335
Real value is  3140 | The predicted value is  3433
Real value is  2639 | The predicted value is  3784
Real value is   658 | The predicted value is   193
Real value is  1125 | The predicted value is  1186
Real value is  1943 | The predicted value is  2426
Real value is   449 | The predicted value is  -969
Real value is   971 | The predicted value is  2054
Real value is   765 | The predicted value is  -145
Real value is 14476 | The predicted value is 14071
```

##### 3.3.2 Gradient Boosting
Builds a strong predictive model by sequentially adding weak learners, typically decision trees, to the ensemble.

```python
from sklearn.ensemble import GradientBoostingRegressor

gradientBoosting_model = GradientBoostingRegressor(random_state=42)
gradientBoosting_model.fit(X_train, y_train)

gradientBoosting_y_pred = gradientBoosting_model.predict(X_test)
gradientBoosting_RMSE = np.sqrt(mean_squared_error(y_test, gradientBoosting_y_pred))
gradientBoosting_MAE = mean_absolute_error(y_test, gradientBoosting_y_pred)

print("Gradient Boosting RMSE:", gradientBoosting_RMSE)
print("Gradient Boosting MAE  :", gradientBoosting_MAE, "\n")

for index, (real_value, predicted_value) in zip(range(10), zip(y_test, gradientBoosting_y_pred)):
    print(f"Real value is {str(round(real_value)).rjust(5)} | The predicted value is {str(round(predicted_value)).rjust(5)}")
```
**Output :**
```markdown
Gradient Boosting RMSE: 624.7616973831518
Gradient Boosting MAE  : 338.49529262139635 

Real value is  3390 | The predicted value is  3304
Real value is  3140 | The predicted value is  2822
Real value is  2639 | The predicted value is  3210
Real value is   658 | The predicted value is   755
Real value is  1125 | The predicted value is   950
Real value is  1943 | The predicted value is  1908
Real value is   449 | The predicted value is   452
Real value is   971 | The predicted value is  1210
Real value is   765 | The predicted value is   777
Real value is 14476 | The predicted value is 16082
```

##### 3.3.3 Decision Tree
It's a tree-like structure where each internal node represents a "test" on an attribute each branch represents the outcome of the test, and each leaf node represents a class label or a numeric value (in regression tasks)

```python
from sklearn.tree import DecisionTreeRegressor

decisionTree_model = DecisionTreeRegressor()
decisionTree_model.fit(X_train, y_train)

decisionTree_y_pred = decisionTree_model.predict(X_test)
decisionTree_RMSE = np.sqrt(mean_squared_error(y_test, decisionTree_y_pred))
decisionTree_MAE = mean_absolute_error(y_test, decisionTree_y_pred)

print("Decision Tree RMSE :", decisionTree_RMSE)
print("Decision Tree MAE  :", decisionTree_MAE, "\n")

for index, (real_value, predicted_value) in zip(range(10), zip(y_test, decisionTree_y_pred)):    
    print(f"Real value is {str(round(real_value)).rjust(5)} | The predicted value is {str(round(predicted_value)).rjust(5)}")
```
**Output :**
```markdown
Decision Tree RMSE : 717.1974514666841
Decision Tree MAE  : 346.60054137664343 

Real value is  3390 | The predicted value is  3780
Real value is  3140 | The predicted value is  2606
Real value is  2639 | The predicted value is  3084
Real value is   658 | The predicted value is   658
Real value is  1125 | The predicted value is   877
Real value is  1943 | The predicted value is  1837
Real value is   449 | The predicted value is   576
Real value is   971 | The predicted value is  1162
Real value is   765 | The predicted value is   661
Real value is 14476 | The predicted value is 18795
```

##### 3.3.4 Random Forest
Builds multiple decision trees during training and outputs the average prediction (regression) of the individual trees. 
It combines the predictions of multiple decision trees to improve generalization and robustness over a single decision tree.

```python
from sklearn.ensemble import RandomForestRegressor

randomForest_model = RandomForestRegressor(n_estimators=200, random_state=42)
randomForest_model.fit(X_train, y_train)

randomForest_y_pred = randomForest_model.predict(X_test)
randomForest_RMSE = np.sqrt(mean_squared_error(y_test, randomForest_y_pred))
randomForest_MAE  = mean_absolute_error(y_test, randomForest_y_pred)

print("Random Forest RMSE :", randomForest_RMSE)
print("Random Forest MAE  :", randomForest_MAE, "\n)

for index, (real_value, predicted_value) in zip(range(1, 20), zip(y_test, randomForest_y_pred)):        
    print(f"Real value is {str(round(real_value)).rjust(5)} | The predicted value is {str(round(predicted_value)).rjust(5)}")
```
**Output :**
```markdown
Random Forest RMSE : 518.8744073629715
Random Forest MAE  : 262.29325709636134

Real value is  3390 | The predicted value is  3362
Real value is  3140 | The predicted value is  2818
Real value is  2639 | The predicted value is  3297
Real value is   658 | The predicted value is   716
Real value is  1125 | The predicted value is   898
Real value is  1943 | The predicted value is  1890
Real value is   449 | The predicted value is   540
Real value is   971 | The predicted value is  1080
Real value is   765 | The predicted value is   667
Real value is 14476 | The predicted value is 16464
Real value is  2496 | The predicted value is  2516
Real value is  6023 | The predicted value is  6267
Real value is   600 | The predicted value is   653
Real value is 16171 | The predicted value is 17368
Real value is   475 | The predicted value is   517
Real value is  5292 | The predicted value is  4725
Real value is  2772 | The predicted value is  3058
Real value is  5599 | The predicted value is  5843
Real value is  4238 | The predicted value is  4329
```

***

#### 3.4 Model Evaluation
Critical step in assessing the performance and effectiveness of a machine learning model
```python
randomForest_model2 = RandomForestRegressor()
forest_scores = cross_val_score(randomForest_model2, X_train, y_train, scoring = "neg_mean_squared_error", cv = 5)
forest_rmse_scores = np.sqrt(np.abs(forest_scores))

for score in forest_rmse_scores:
    print(round(score, 3))
```
***

#### 3.5 Model Fine-Tune
Process of optimizing the hyperparameters of a machine learning model to improve its performance on the validation or test dataset
```python
param_grid = {
    'n_estimators': [50, 100],
    'max_features': [2, 3],
    'max_depth'   : [None],
    'min_samples_split' : [2, 5],
    'min_samples_leaf'  : [1, 2],
}

randomForest_model3 = RandomForestRegressor(random_state = 42)

grid_search = GridSearchCV(randomForest_model3, param_grid, cv = 5, scoring = 'neg_mean_squared_error', return_train_score = True)
grid_search.fit(X_train, y_train)
```
- Grid Search: Exhaustively searching through a specified set of hyperparameters to find the combination that yields the best performance
```python
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"],cvres["params"]):
    print(round(np.sqrt(np.abs(mean_score)), 2), params)
```
**Output :**
```markdown
559.13 {'max_depth': None, 'max_features': 2, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}
555.52 {'max_depth': None, 'max_features': 2, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
559.31 {'max_depth': None, 'max_features': 2, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 50}
557.99 {'max_depth': None, 'max_features': 2, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}
565.79 {'max_depth': None, 'max_features': 2, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 50}
563.62 {'max_depth': None, 'max_features': 2, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}
566.91 {'max_depth': None, 'max_features': 2, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 50}
565.55 {'max_depth': None, 'max_features': 2, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100}
540.74 {'max_depth': None, 'max_features': 3, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}
539.48 {'max_depth': None, 'max_features': 3, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
541.43 {'max_depth': None, 'max_features': 3, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 50}
539.31 {'max_depth': None, 'max_features': 3, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}
544.91 {'max_depth': None, 'max_features': 3, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 50}
542.65 {'max_depth': None, 'max_features': 3, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}
545.04 {'max_depth': None, 'max_features': 3, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 50}
543.18 {'max_depth': None, 'max_features': 3, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100}
```
- Features importance : Used to identify the most influential features in the model
```python
feature_importances = grid_search.best_estimator_.feature_importances_

# Plotting feature importances
plt.figure(figsize=(10, 6))
plt.bar(X.columns, feature_importances)
plt.xlabel('Features')
plt.ylabel('Feature Importance')
plt.show()
```
**insights :**
- The most influential features in the model are `vol` and `carat`

***
#### 3.6 Model predictions
- Retrieves the best model obtained from a grid search
- Applies this model to make predictions on the `submission_data`

```python
best_model = grid_search.best_estimator_
best_predictions = best_model.predict(submission_data)
```

***

####  3.7 Submission Dataset (For Competition)
The dataset to be submitted it should contain only 2 columns, 1st one is the IDs, 2nd one is the predictions
So we need to concatenate the test_df IDs with the model predictions
```python
data_for_sub = pd.DataFrame({"ID":test_IDs, "price":best_predictions2})
data_for_sub.to_csv("Submission RF.csv", index=False)
```

### 4 Summary
This project focused on developing a machine learning regression model to predict diamond prices using a dataset containing various attributes of nearly 54,000 diamonds. Key steps and findings include:

- Data Exploration and Preparation: The dataset was thoroughly explored, and essential preprocessing steps were taken, including data cleaning and feature engineering. Significant attributes such as carat weight, cut quality, color grade, and clarity were identified as primary predictors of diamond prices.

- Modeling: Several regression models were tested, including linear regression, decision trees, random forests, and gradient boosting. Random forests emerged as the most effective model, providing the highest prediction accuracy.

- Evaluation: The final model demonstrated strong performance, accurately capturing the relationships between diamond attributes and prices, making it a valuable tool for consumers and retailers in the diamond market.

