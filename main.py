import pandas as pd

# for the titles because I'm lazy and I don't feel like looping
file = pd.read_csv('/Users/eunjikim/PycharmProjects/rRelationships/batched_comments_1m/comments_20101231-20120507.csv')

print(file.head()['selftext'])

# df = pd.read_csv('/Users/eunjikim/PycharmProjects/rRelationships/allcomments.csv')

# checking the number of empty rows in th csv file
# print(df.isnull().sum())

# Dropping empty rows
# modifiedDF = df.dropna()

# Saving it to the csv file
# modifiedDF.to_csv('allcomments_noblanks.csv', index = False)
