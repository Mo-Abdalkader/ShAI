# Documentation

## Content Table
1. Problem Statement
2. Data Description
3. Methodology
   1. Data Preprocessing
   2. Data Visualization
   3. Model Selection & Training
   4. Model Evaluation
   5.  Submission Dataset (For Competition)
4. Summary


### 1. Problem Statement
The objective of this project is to develop a predictive model that accurately estimates the price of diamonds based on their attributes. Given the significant variability in diamond prices due to differences in physical characteristics and quality, the goal is to utilize machine learning techniques to identify patterns and relationships within the dataset that can predict the price of a diamond. This will aid consumers and retailers in making informed decisions regarding diamond purchases and sales.

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

