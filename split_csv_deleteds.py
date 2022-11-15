import ssl
import os
import pandas as pd


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


dirstr = '/Users/eunjikim/PycharmProjects/rRelationships/batched_comments_1m/'
directory = os.fsencode(dirstr)


for file in os.listdir(directory):
    filename = os.fsdecode(file)
    # call on every file ending in csv
    if filename.endswith('0416.csv'):
        # open as a dataframe
        df = pd.read_csv(dirstr + filename)
        # print(df.columns)
        # initiating new dfs with col names for deleted / not deleted posts
        # df for deleted posts / comments: '[removed]', for comments: '[deleted]'
        df_deleted = df[df['selftext'] == '[deleted]']  # df for deleted posts
        df_posts = df[df['selftext'] != '[deleted]']  # df for reg posts

        # create new file with new name into other directory
        df_deleted.to_csv(dirstr + '/comments_deleted/deleted_' + filename)
        df_posts.to_csv(dirstr + '/comments_not_deleted/not_deleted_' + filename)
