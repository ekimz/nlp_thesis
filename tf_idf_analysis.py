import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


# from nltk.tokenize import word_tokenize
# import math
# from statistics import *

dirstr = '/Users/eunjikim/PycharmProjects/rRelationships/batched_comments_1m/comments_not_deleted/'
# dirstr = '/Users/eunjikim/PycharmProjects/rRelationships/batched_comments_1m/' for comments
directory = os.fsencode(dirstr)
# print(list(os.listdir(directory)))

# create empty tf-idf summary df
header = ['word', 'frequency']
tfidf_summary_df = pd.DataFrame(columns=header)
# print(tfidf_summary_df)
# row_values = []



# the for loop, as per usual

for file in sorted(os.listdir(directory)):

    filename = os.fsdecode(file)

    # call on every file ending in csv
    if filename.endswith('.csv'):
        # open file as a dataframe
        df = pd.read_csv(dirstr + filename)

        # put together all self text / comments as a corpus
        # ALL as one string; doesn't matter for posts within an era
        file_corpus = ' '.join(df['selftext'])

        # run tf-idf on the file corpus
        count = CountVectorizer()
        word_count = count.fit_transform(file_corpus)
        # print(word_count)
        # word_count.shape
        # print(word_count.toarray())
        tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        tfidf_transformer.fit(word_count)
        #df_idf = pd.DataFrame(tfidf_transformer.idf_,
         #                     index=count.get_feature_names(),
          #                    columns=["idf_weights"])
        # inverse document frequency
        # df_idf.sort_values(by=['idf_weights'])
        # tf-idf
        tf_idf_vector = tfidf_transformer.transform(word_count)
        feature_names = count.get_feature_names()

        first_document_vector = tf_idf_vector[1]
        df_tfidf = pd.DataFrame(first_document_vector.T.todense(),
                                index=feature_names,
                                columns=["tfidf"])

        df_tfidf.sort_values(by=["tfidf"], ascending=False)
        print(df_tfidf.head(10))

        # & then add the frequency to the according column
        era = filename[15:-4]
        print(f"Adding result for era: {era}")


# summary_df.to_csv(dirstr + 'sentiment_scores_summary.csv')
print(summary_df.head())
summary_df.to_csv('./comments_not_deleted_sentiment_summary.csv', index=False)
