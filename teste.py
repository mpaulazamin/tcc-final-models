# Importing Pandas package
import pandas as pd

# Creating a Dictionary
d = {
    'a': ['120,00', '42,00', '18,00', '23,00'],
    'b': ['51,23', '18,45', '28,90', '133,00']
}

# Creating a dataframe
df = pd.DataFrame(d)

# Display Original dataframe
print("Created Dataframe:\n",df,"\n")

# Replacing , with .
df['a'] = df['a'].replace(',','.',regex=True)
df['b'] = df['b'].replace(',','.',regex=True)

# Converting the data type
df['a'] = df['a'].astype(float)
df['b'] = df['b'].astype(float)

# Display modified DataFrame
print("Modified DataFrame:\n",df)