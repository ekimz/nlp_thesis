import os
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
# nltk.download('stopwords') # when you don't have
# nltk.download('punkt') # when you don't have
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import gensim
from gensim.utils import simple_preprocess
print(stopwords.words('english'))
stopwords = stopwords.words('english')
from nltk.tokenize import word_tokenize

# import math
# from statistics import *
# from gensim.parsing.preprocessing import remove_stopwords

dirstr = '/Users/eunjikim/PycharmProjects/rRelationships/batched_comments_1m/comments_not_deleted/'
# dirstr = '/Users/eunjikim/PycharmProjects/rRelationships/batched_comments_1m/' for comments
directory = os.fsencode(dirstr)
# print(list(os.listdir(directory)))


# create empty tf-idf summary df
header = ['word', 'frequency']
tfidf_summary_df = pd.DataFrame(columns=header)
# print(tfidf_summary_df)
# row_values = []

# just the tf-idf to see if this shit actually works

df = pd.read_csv(
    '/Users/eunjikim/PycharmProjects/rRelationships/batched_comments_1m/comments_not_deleted/not_deleted_sentiment_comments_20150124-20150524.csv')


# three lines borrowed from elsewhere
corpora = df.drop_duplicates(subset=['selftext'])  # remove duplicate comments
corpora = corpora.dropna(subset=['selftext'])  # remove NaN
corpora = corpora[corpora['selftext'].str.contains('removed|wiki') == False] # remove any comments including "submission removed"

corpora = corpora['selftext'].str.lower() # was corpora = df[...]
corpora['sans_stopwords'] = corpora.apply(lambda x: ' '.join(
    [word for word in x.split() if word not in (stopwords)]
))

# print(corpora['sans_stopwords'])

# isolate 'selftext' column
file_corpus = ' '.join(corpora['sans_stopwords'])


tfIdfVectorizer = TfidfVectorizer(ngram_range=(1,2), use_idf=True)
tfIdf = tfIdfVectorizer.fit_transform([file_corpus])
df = pd.DataFrame(tfIdf[0].T.todense(),
                  index=tfIdfVectorizer.get_feature_names_out(),
                  columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)
print(df)