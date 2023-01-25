import re
import ssl
import os
import pandas as pd
import nltk

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

directory = os.fsencode('/Users/eunjikim/PycharmProjects/rRelationships/batched_comments_1m/comments_not_deleted/')
dirstr = '/Users/eunjikim/PycharmProjects/rRelationships/batched_comments_1m/comments_not_deleted/'

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
        preprocess_tag = [text_prep(i) for i in df['selftext']]  # df[...] is the content you wan analyse
        df['preprocess'] = preprocess_tag
        df['total_len'] = df['preprocess'].map(lambda x: len(x))

        # open necessary files for sentiment analysis
        file_pos = open('/Users/eunjikim/PycharmProjects/rRelationships/positive-words.txt',
                        'r', encoding='ISO-8859-1')
        pos_words = file_pos.read().split()

        file_neg = open('/Users/eunjikim/PycharmProjects/rRelationships/negative-words.txt',
                        'r', encoding='ISO-8859-1')
        neg_words = file_neg.read().split()

        # count positive words
        num_pos = df['preprocess'].map(lambda x: len([i for i in x if i in pos_words]))
        df['pos_count'] = num_pos
        # count negative words
        num_neg = df['preprocess'].map(lambda x: len([i for i in x if i in neg_words]))
        df['neg_count'] = num_neg

        df['sentiment'] = round((df['pos_count'] - df['neg_count']) / df['total_len'], 2)

        # create new file with new name I suppose
        df.to_csv(dirstr + '_' + filename) # sentiment_
        continue
    else:
        continue
