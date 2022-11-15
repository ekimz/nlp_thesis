import os
import pandas as pd

directory = os.fsencode('/Users/eunjikim/PycharmProjects/rRelationships/batched_posts_100k/')
dirstr = '/Users/eunjikim/PycharmProjects/rRelationships/batched_posts_100k/'

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    # call on every file ending in csv
    if filename.endswith('.csv'):
        # open as a dataframe
        df = pd.read_csv(dirstr + filename)
        # create a new name for your text file
        newname = dirstr + filename[:-4] + '.txt'
        # use wanted column name to get corpora in text form
        df['title'].to_csv(newname, sep="\n", index=False) # replace 'title' for other content
        continue
    else:
        continue

