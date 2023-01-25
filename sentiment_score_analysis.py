import os
import pandas as pd
import numpy as np  # for declaring an array or simply use list
import math
from statistics import *

dirstr = '/Users/eunjikim/PycharmProjects/rRelationships/batched_comments_1m/comments_not_deleted/'
# dirstr = '/Users/eunjikim/PycharmProjects/rRelationships/batched_comments_1m/' for comments
directory = os.fsencode(dirstr)
header = ['batch_date', 'min', 'max', 'mean', 'median', 'std', 'var']

# print(list(os.listdir(directory)))

summary_df = pd.DataFrame(columns=header)
# print(summary_df)
row_values = []

for file in sorted(os.listdir(directory)):
    filename = os.fsdecode(file)

    # call on every file ending in csv
    if filename.endswith('.csv'):
        # open file as a dataframe
        df = pd.read_csv(dirstr + filename)
        # subset dataframe to show only the sentiment scores
        # just 'sentiment' for comments, 'selftext_sentiment' for posts
        sentiment_scores_df = df[['sentiment']]
        # print(type(sentiment_scores_df))

        # create sentiment score summary, add to to proper column
        era = filename[15:-4]
        print(f"Adding result for era: {era}")
        row_values = [era,
                      sentiment_scores_df.min()[0],
                      sentiment_scores_df.max()[0],
                      sentiment_scores_df.mean()[0],
                      sentiment_scores_df.median()[0],
                      # sentiment_scores_df.mode()[0],
                      sentiment_scores_df.std()[0],
                      sentiment_scores_df.var()[0]]

        # add row_values to summary_df
        summary_df.loc[len(summary_df)] = row_values
        print(f"result is length: {len(summary_df)}")

# summary_df.to_csv(dirstr + 'sentiment_scores_summary.csv')
print(summary_df.head())
summary_df.to_csv('./comments_not_deleted_sentiment_summary.csv', index=False)
