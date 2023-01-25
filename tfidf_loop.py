import os
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet


# import math
# from statistics import *
# from gensim.parsing.preprocessing import remove_stopwords

# dirstr = '/Users/eunjikim/PycharmProjects/rRelationships/batched_posts_100k/posts_not_deleted/'  # for post titles
# dirstr = '/Users/eunjikim/PycharmProjects/rRelationships/batched_posts_100k/posts_not_deleted/'  # for posts
dirstr = '/Users/eunjikim/PycharmProjects/rRelationships/batched_comments_1m/comments_not_deleted/'  # for comments

directory = os.fsencode(dirstr)
# print(list(os.listdir(directory)))


# print(stopwords.words('english'))
stopwords = stopwords.words('english')
stopwords.extend(['from', 'subject', 're', 'edu', 'use', 'im', 'hes', 'shes', 'like', 'youre', 'youll',
                  'subreddit', 'moderator', 'https', 'gt'])
lemmatizer = WordNetLemmatizer()


# create empty tf-idf summary df
header = ['word', 'frequency']
tfidf_summary_df = pd.DataFrame(columns=header)
# print(tfidf_summary_df)
# row_values = []


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def text_prep(x):
    corp = str(x).lower()
    corp = re.sub('[^a-zA-Z]+', ' ', corp).strip()
    tokens = word_tokenize(corp)
    words = [t for t in tokens if t not in stopwords]
    # lemmatize = [lemma.lemmatize(w) for w in words] old shit
    pos_tagged = nltk.pos_tag(words)
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            # else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    lemmatized_sentence = " ".join(lemmatized_sentence)

    return lemmatized_sentence


for file in sorted(os.listdir(directory)):
    filename = os.fsdecode(file)
    # call on every file ending in csv
    if filename.endswith('.csv'):
        # open as a dataframe
        df = pd.read_csv(dirstr + filename)
        # era = filename[27:-4]  # posts
        era = filename[31:-4]  # comments
        # print(era)

        preprocessed = [text_prep(i) for i in df['selftext']]  # df[...] is the content you warn analyse; posts/comments
        # preprocessed = [text_prep(i) for i in df['title']]  # for titles
        # print(preprocessed)

        df['preprocess_posts'] = preprocessed
        df['string_pp'] = [''.join(map(str, l)) for l in df['preprocess_posts']]  # for titles

        # print(df.head(10))

        # three lines borrowed from elsewhere
        corpora = df.drop_duplicates(subset=['string_pp'])  # remove duplicate comments
        corpora = corpora.dropna(subset=['string_pp'])  # remove NaN
        corpora = corpora[
            corpora['string_pp'].str.contains('removed|wiki') == False]  # remove comments incl "submission removed"

        # preprocessed so don't need these for now
        # corpora = corpora['preprocess_posts'].str.lower()  # was corpora = df[...]
        # corpora['sans_stopwords'] = corpora.apply(lambda x: ' '.join(
        #     [word for word in x.split() if word not in (stopwords)]
        # ))

        # print(corpora['sans_stopwords'])

        # isolate 'selftext' column
        file_corpus = ' '.join(corpora['string_pp'])
        # print(file_corpus)

        tfIdfVectorizer = TfidfVectorizer(ngram_range=(1, 2), use_idf=True)
        tfIdf = tfIdfVectorizer.fit_transform([file_corpus])
        df = pd.DataFrame(tfIdf[0].T.todense(),
                          index=tfIdfVectorizer.get_feature_names_out(),
                          columns=["TF-IDF"])
        df = df.sort_values('TF-IDF', ascending=False)
        df.insert(0, 'term_pp', df.index)
        # print(df)

        # create new file with new name I suppose
        # df.to_csv(dirstr + 'tfidf' + era + '.csv')  # for posts, titles
        df.to_csv(dirstr + 'tfidf_' + era + '.csv')  # for comments
        print('saved ' + era + ' file UwU')
