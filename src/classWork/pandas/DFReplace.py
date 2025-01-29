import pandas as pd

# Create a sample DataFrame

df = pd.DataFrame({

    'A': [1, 2, 3, 4, 1],

    'B': [5, 6, 7, 8, 5],

    'C': ['foo', 'fool', 'bazz', 'bar', 'foo']

})

# Initial DataFrame

print("Initial DataFrame:")

print(df)

print()

# Replace a single value (1 with 10)

df = df.replace(1, 10)

print("# After replacing 1 with 10:")

print(df)

# Replace multiple values (2 and 3 with 20 and 30, #respectively)

df = df.replace({2: 20, 3: 30})

print("# After replacing 2 with 20 and 3 with 30:")

print(df)

# Replace using a list (4 and 5 with 40 and 50, #respectively)

df = df.replace([4, 5], [40, 50])

print("# After replacing 4 with 40 and 5 with 50:")

print(df)

# Replace using regular expressions (all occurrences of #'foo' with 'boo')

df = df.replace(to_replace=r'^foo', value='boo', regex=True)

print("# After replacing all occurrences of 'foo' with 'boo':")

print(df)

# Replace values in a specific column (column 'C', replacing #'boo' with 'foo')

df['C'] = df['C'].replace('boo', 'foo')

print("# After replacing 'boo' with 'foo' in column 'C':")

print(df)