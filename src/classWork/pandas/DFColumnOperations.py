import pandas as pd

import numpy as np

# Creating a sample DataFrame

data = {

    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],

    'Age': [25, 30, 35, 40, 29],

    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],

    'Salary': [70000, 80000, 120000, 110000, 95000]

}

df = pd.DataFrame(data)

# Basic Arithmetic Operations

df['Salary_increase'] = df['Salary'] * 1.10

# Increase salary by 10%

# Statistical Calculations

mean_age = df['Age'].mean()

median_salary = df['Salary'].median()

std_salary = df['Salary'].std()

# Aggregation Operations

grouped_data = df.groupby('City')['Salary'].mean()

# Mean salary by city

# Applying Functions

df['Age_squared'] = df['Age'].apply(lambda x: x**2)

# Logical Operations

df['High_salary'] = df['Salary'] > 100000

# Transformation Operations

df['Log_salary'] = np.log(df['Salary'])

# String Operations

df['City_upper'] = df['City'].str.upper()

# Displaying the DataFrame with new calculations

print(df)

print("\nMean age:", mean_age)

print("Median salary:", median_salary)

print("Standard deviation of salary:", std_salary)

print("\nMean salary by city:\n", grouped_data)