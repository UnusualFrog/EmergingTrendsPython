import pandas as pd

# Create a sample DataFrame with a custom index

df = pd.DataFrame({

    'ID': [101, 102, 103, 104],

    'Name': ['Alice', 'Bob', 'Charlie', 'David'],

    'Age': [25, 30, 35, 40],

    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']

}).set_index('ID')

# Initial DataFrame with 'ID' as the index

print("Initial DataFrame with 'ID' as the index:")

print(df)

print()

# Reset the index

df_reset = df.reset_index()

print("# After resetting the index:")

print(df_reset)

print()

# Create a sample DataFrame with a multi-index

df_multi = df.set_index(['City', 'Name'])

# Initial DataFrame with multi-index

print("Initial DataFrame with multi-index:")

print(df_multi)

print()

# Reset the multi-index

df_multi_reset = df_multi.reset_index()

print("# After resetting the multi-index:")

print(df_multi_reset)

print()