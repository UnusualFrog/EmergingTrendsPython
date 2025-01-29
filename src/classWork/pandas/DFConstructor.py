import pandas as pd

# Create a DataFrame using the DataFrame constructor

data = {

    'Name': ['John', 'Mary', 'Peter'],

    'Age': [20, 25, 30],

    'City': ['New York', 'Boston', 'Chicago']

}

df = pd.DataFrame(data)

# Print the DataFrame

print(df)