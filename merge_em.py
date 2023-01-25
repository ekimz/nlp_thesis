import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


dirstr = '/Users/eunjikim/PycharmProjects/rRelationships/batched_posts_100k/tfidf_posts/'  # for posts
# dirstr = '/Users/eunjikim/PycharmProjects/rRelationships/batched_posts_100k/tfidf_titles/'  # for post titles
# dirstr = '/Users/eunjikim/PycharmProjects/rRelationships/batched_comments_1m/tfidf_comments/'  # for comments
directory = os.fsencode(dirstr)


# create empty tf-idf summary df
header = ['term_pp']
tfidf_summary_df = pd.DataFrame(columns=header)


for file in sorted(os.listdir(directory)):
    filename = os.fsdecode(file)
    # call on every file ending in csv
    if filename.endswith('.csv'):
        # open as a dataframe
        df = pd.read_csv(dirstr + filename)

        df = df.drop(columns='Unnamed: 0')
        era = filename[7: -4]  # for posts
        # era = filename[6: -4]  # for post titles, comments

        # era = filename[0: -4]  # for comments
        df.rename(columns={'TF-IDF': era}, inplace=True)

        # and this is the MOMENT OF TRUTH UWU

        tfidf_summary_df = pd.merge(tfidf_summary_df,
                                    df[['term_pp', era]],  # df[['term', era]],  # on='term'  # posts & comments
                                    on='term_pp',  # post titles
                                    how='outer')
        print(f"Added result for era: {era}")

print(tfidf_summary_df)
tfidf_summary_df.to_csv('./tfidf_posts_summary.csv', index=False)  # for posts
# tfidf_summary_df.to_csv('./tfidf_titles_summary.csv', index=False)  # for posts
# tfidf_summary_df.to_csv('./tfidf_comments_summary.csv', index=False) # for comments
