import pandas as pd

# Reading the DataFrame from the pickle file

pickle_filename = 'dataframe.pkl'

restored_df = pd.read_pickle(pickle_filename)

# Displaying the restored DataFrame

print(restored_df)