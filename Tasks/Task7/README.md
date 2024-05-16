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


### 


```markdown
price -----------------------------------------------------> $326 : $18,823

carat weight ---------------------------------------------->  0.2 : 5.01
cut quality -----------------------------------------------> (Fair, Good, Very Good, Premium, Ideal)
color 'diamond colour' ------------------------------------> from J (worst) to D (best)
                                                                  d e f g h i j
                                                                  6 5 4 3 2 1 0
clarity "how clear the diamond is" ------------------------> (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))

x 'length' ------------------------------------------------> 0 : 10.74 mm
y 'width'  ------------------------------------------------> 0 : 58.90 mm
z 'depth'  ------------------------------------------------> 0 : 31.80 mm

depth 'total depth percentage' ----------------------------> 43 : 79 (%)
       = z / mean(x, y) = 2 * z / (x + y)
                             
table 'width of top of diamond relative to widest point' --> 43 : 95
```