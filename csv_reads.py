import pandas as pd
import math


def to_csv_batch(src_csv, dst_dir, size=1000000, index=False):
    # Read source csv
    df = pd.read_csv(src_csv, low_memory=False)

    # Initial values
    low = 0
    high = size

    # Loop through batches
    for i in range(math.ceil(len(df) / size)):

        fname = dst_dir + '/tfidf_comments_' + str(i + 1) + '.csv'
        df[low:high].to_csv(fname, index=index)
        print('we doin da ting for' + fname)

        # Update selection
        low = high
        if (high + size < len(df)):
            high = high + size
        else:
            high = len(df)



to_csv_batch('/Users/eunjikim/PycharmProjects/rRelationships/tfidf_comments_summary.csv',
             '/Users/eunjikim/PycharmProjects/rRelationships/batched_comments_1m/tfidf_split')
