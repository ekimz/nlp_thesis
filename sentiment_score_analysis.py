import os
import pandas as pd
import numpy as np  # for declaring an array or simply use list
import math
from statistics import *

dirstr = '/Users/eunjikim/PycharmProjects/rRelationships/batched_comments_1m/sentiment_comments/'
directory = os.fsencode(dirstr)
header = ['batch_date', 'min', 'max', 'mean', 'median', 'mode', 'std', 'var']

summary_df = pd.DataFrame(columns=header)
# print(summary_df)
row_values = []

for file in os.listdir(directory):
    filename = os.fsdecode(file)

    # call on every file ending in csv
    if filename.endswith('.csv'):
        # open file as a dataframe
        df = pd.read_csv(dirstr + filename)
        # subset dataframe to show only the sentiment scores
        sentiment_scores_df = df[['sentiment']]

        # create sentiment score summary, add to to proper column
        row_values = [filename[19:-4], sentiment_scores_df.min(), sentiment_scores_df.max(),
                      sentiment_scores_df.mean(), sentiment_scores_df.median(), sentiment_scores_df.mode(),
                      sentiment_scores_df.std(), sentiment_scores_df.var()]

        # add row_values to sentiment_scores_df
        summary_df.loc[len(df)] = row_values
        # update summary_df & keeps until the end
        print(summary_df)
        continue

    else:
        # save new file with new name I suppose
        summary_df.to_csv(dirstr + 'sentiment_scores_summary')
        continue
