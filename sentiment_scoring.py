import re
import ssl
import os
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('omw-1.4')

directory = os.fsencode('/Users/eunjikim/PycharmProjects/rRelationships/batched_posts_100k/')
dirstr = '/Users/eunjikim/PycharmProjects/rRelationships/batched_posts_100k/'

lemma = WordNetLemmatizer()
stop_words = stopwords.words('english')


def text_prep(x):
    corp = str(x).lower()
    corp = re.sub('[^a-zA-Z]+', ' ', corp).strip()
    tokens = word_tokenize(corp)
    words = [t for t in tokens if t not in stop_words]
    lemmatize = [lemma.lemmatize(w) for w in words]

    return lemmatize


for file in os.listdir(directory):
    filename = os.fsdecode(file)
    # call on every file ending in csv
    if filename.endswith('.csv'):
        # open as a dataframe
        df = pd.read_csv(dirstr + filename)
        # preprocess title
        preprocess_tag = [text_prep(i) for i in df['title']]  # df[...] is the content you wan analyse
        df['preprocess_title'] = preprocess_tag
        df['total_len_title'] = df['preprocess_title'].map(lambda x: len(x))
        # preprocess post
        preprocess_tog = [text_prep(i) for i in df['selftext']]
        df['preprocess_selftext'] = preprocess_tog
        df['total_len_selftext'] = df['preprocess_selftext'].map(lambda x: len(x))

        # open necessary files for sentiment analysis
        file = open('/Users/eunjikim/PycharmProjects/rRelationships/negative-words.txt',
                    'r', encoding='ISO-8859-1')
        neg_words = file.read().split()
        ile = open('/Users/eunjikim/PycharmProjects/rRelationships/positive-words.txt',
                   'r', encoding='ISO-8859-1')
        pos_words = file.read().split()

        # count positive words
        num_pos_title = df['preprocess_title'].map(lambda x: len([i for i in x if i in pos_words]))
        df['pos_count_title'] = num_pos_title
        # count negative words
        num_neg_title = df['preprocess_title'].map(lambda x: len([i for i in x if i in neg_words]))
        df['neg_count_title'] = num_neg_title

        df['title_sentiment'] = round((df['pos_count_title'] - df['neg_count_title']) / df['total_len_title'], 2)

        # count positive words
        num_pos_selftext = df['preprocess_selftext'].map(lambda x: len([i for i in x if i in pos_words]))
        df['pos_count_selftext'] = num_pos_selftext
        # count negative words
        num_neg_selftext = df['preprocess_selftext'].map(lambda x: len([i for i in x if i in neg_words]))
        df['neg_count_selftext'] = num_neg_selftext

        df['selftext_sentiment'] = round((df['pos_count_selftext'] - df['neg_count_selftext']) / df['total_len_selftext'], 2)

        # create new file with new name I suppose
        df.to_csv(dirstr + 'sentiment_' + filename)
        continue
    else:
        continue

# df = pd.read_csv('/Users/eunjikim/PycharmProjects/rRelationships/batched_posts_100k/posts_20080723-20120424.csv')


# separate those with positive and negative sentiment scores & run LDA
# you don't need to save these to new files because you should be able
# to call on them to run LDA by using the below parameters

# neg_df = df[df['sentiment'] < 0]
# zil_df = df[df['sentiment'] == 0]
# pos_df = df[df['sentiment'] > 0]
