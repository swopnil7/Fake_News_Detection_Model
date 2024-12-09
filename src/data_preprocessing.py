import pandas as pd
import numpy as np

# Load the datasets
true = pd.read_csv('data/raw/True.csv')
fake = pd.read_csv('data/raw/Fake.csv')

# Display the first few rows of each dataset
print(true.head())
print(fake.head())

# Assign labels to the datasets
true['label'] = 1
fake['label'] = 0

# Display the updated dataframes
print(true.head())
print(fake.head())

# Concatenate the datasets
news = pd.concat([fake, true], axis=0)

# Display the concatenated dataframe
print(news.head())

# Check for null values
print(news.isnull().sum())

# Drop unnecessary columns
news = news.drop(['title', 'subject', 'date'], axis=1)

# Display the updated dataframe
print(news.head())

# Shuffle the dataset
news = news.sample(frac=1)

# Display the shuffled dataframe
print(news.head())

# Reset index
news.reset_index(inplace=True)

# Display the dataframe after resetting the index
print(news.head())

# Drop the index column
news.drop(['index'], axis=1, inplace=True)

# Display the final dataframe
print(news.head())
