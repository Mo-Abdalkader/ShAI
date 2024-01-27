# Summary of Insights Report

## Tasks

### 1. Basic Data Exploration
- Identify the number of rows and columns in the dataset.
- Determine the data types of each column.
- Check for missing values in each column.

### 2. Descriptive Statistics
- Calculate basic statistics: mean, median, mode, minimum, and maximum salary.
- Determine the range of salaries.
- Find the standard deviation.

### 3. Data Cleaning
- Handle missing data using a suitable method and explain why you chose it.

### 4. Basic Data Visualization
- Create histograms or bar charts to visualize the distribution of salaries.
- Use pie charts to represent the proportion of employees in different departments.

### 5. Grouped Analysis
- Group the data by one or more columns.
- Calculate summary statistics for each group.
- Compare the average salaries across different groups.

### 6. Simple Correlation Analysis
- Identify any correlation between salary and another numerical column.
- Plot a scatter plot to visualize the relationship.

### 8. Summary of Insights

1. **Salary Distribution:**
   The histogram analysis revealed that the majority of employees have salaries in the range of $60,000 to $80,000.

2. **Department Proportions:**
   The pie chart illustrated the proportion of employees in different departments.

3. **Grouped Analysis:**
   Grouping the data by department, the average salary for each department was calculated.

4. **Correlation Analysis:**
   A scatter plot and correlation coefficient were used to examine the relationship between salary and years of experience. 
   The correlation was positive, suggesting that as years of experience increased, so did the salary.

5. **Data Cleaning:**
   A note highlighted that 'Benefits' and 'OtherPay' columns had many missing values but did not affect the total result, so they were replaced with zeros.

6. **Conditions for Deletion:**
   Rows with the sum of ('BasePay', 'OvertimePay', 'OtherPay', 'Benefits') less than or equal to zero were identified as useless and recommended for deletion.
