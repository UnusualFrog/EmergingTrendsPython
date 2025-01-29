import pandas as pd

# Creating a sample DataFrame

data = {

    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],

    'Age': [25, 30, 35, 40, 29],

    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],

    'Salary': [70000, 80000, 120000, 110000, 95000]

}

df = pd.DataFrame(data)

# Using describe() method

print("Descriptive statistics using describe():")

print(df.describe())

# Using other statistical methods

print("\nSum of ages:")

print(df['Age'].sum())

print("\nMean of salaries:")

print(df['Salary'].mean())

print("\nMedian of ages:")

print(df['Age'].median())

print("\nStandard deviation of salaries:")

print(df['Salary'].std())

print("\nMinimum age:")

print(df['Age'].min())

print("\nMaximum salary:")

print(df['Salary'].max())

print("\nIndex of minimum age:")

print(df['Age'].idxmin())

print("\nIndex of maximum salary:")

print(df['Salary'].idxmax())

print("\nMode of cities:")

print(df['City'].mode())

print("\nAbsolute values of salaries:")

print(df['Salary'].abs())

# Calculate mean absolute deviation manually

mean_age = df['Age'].mean()

mad_ages = (df['Age'] - mean_age).abs().mean()

print("\nMean absolute deviation of ages:")

print(mad_ages)

print("\nValue at 50th percentile (quantile) of salaries:")

print(df['Salary'].quantile(0.5))

print("\nSkewness of ages:")

print(df['Age'].skew())

print("\nKurtosis of salaries:")

print(df['Salary'].kurt())