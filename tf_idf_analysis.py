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
# header = ['batch_date', 'min', 'max', 'mean', 'median', 'std', 'var']
# print(list(os.listdir(directory)))
# summary_df = pd.DataFrame(columns=header)
# print(summary_df)
# row_values = []


# here is the code for tf-idf
# using the count vectorizer

count = CountVectorizer()
word_count = count.fit_transform(_e_d_i_t_text_h_e_r_e_)
# print(word_count)
# word_count.shape
# print(word_count.toarray())
tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(word_count)
df_idf = pd.DataFrame(tfidf_transformer.idf_,
                      index=count.get_feature_names(),
                      columns=["idf_weights"])
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


create tf-idf as a new df
and then in the initated df, sort/organize the created df with if/else statement

# try making the if-else anyway bc what do you have to lose? nothing but time
for tf_word, tfidf_calculation in tfidf_df:
    add new column, name = filename[15:-4]
    if tf_word in main_df_word_col:
        main_df_tfidf_col(iloc_that_row) == tfidf_calculation
    else:
        append(bottom of the column) tf_word
        main_df_tfidf_col(iloc new row) == tfidf_calculation

# the for loop, as per usual

for file in sorted(os.listdir(directory)):
    filename = os.fsdecode(file)

    # call on every file ending in csv
    if filename.endswith('.csv'):
        # open file as a dataframe
        df = pd.read_csv(dirstr + filename)
        # put together all of the self text / comment as a corpus
        file_corpus = df['selftext'].values.tolist()

        # run tf-idf on the file corpus
        aaaa

        # this would prob be best as a dict(?) so that you can
        # run the tf-idf the first time
        # then, when you run it the second time, you need to find whether
        # the word that is there matches any of the ones that are already in the col
        # if they are, you can add that to the new col for the tf-idf for that era
        # if they are not, you need to create a new row with that word in the first col
        # & then add the frequency to the according column
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
