import pandas as pd

# Create a sample DataFrame

df = pd.DataFrame({

    'ID': [101, 102, 103, 104],

    'Name': ['Alice', 'Bob', 'Charlie', 'David'],

    'Age': [25, 30, 35, 40],

    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']

})

# Initial DataFrame

print("Initial DataFrame:")

print(df)

print()

# Set the 'ID' column as the index

df_indexed = df.set_index('ID')

print("# After setting 'ID' as the index:")

print(df_indexed)

print()

# Set multiple columns as the index

df_multi_indexed = df.set_index(['City', 'Name'])

print("# After setting 'City' and 'Name' as the multi-index:")

print(df_multi_indexed)

print()