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
