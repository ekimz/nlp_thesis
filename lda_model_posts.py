# import modules
import pandas as pd
# example of starting a child process only in __main__
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import gensim
from gensim.utils import simple_preprocess
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
import gensim.corpora as corpora
from pprint import pprint


def main():
    # working with just one of the files for now
    dirstr = '/Users/eunjikim/PycharmProjects/rRelationships/'
    filename_posts = 'batched_comments_1m/comments_not_deleted/not_deleted_sentiment_comments_20220328-20221001.csv'
    # 20220328-20221001
    # not_deleted_sentiment_comments_20220328-20221001

    # dataframe
    df = pd.read_csv(dirstr + filename_posts)
    print(df.columns)

    # print(df.columns) # when checking for column names
    # drop unnecessary columns for posts
    titles = df.drop(columns=['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'subreddit',
        'score', 'permalink', 'id', 'preprocess',
       'total_len', 'pos_count', 'neg_count', 'sentiment']
                                             )  # keep selftext & created_utc

    titles = titles.drop_duplicates(subset=['selftext'], keep=False)  # remove duplicate titles
    titles = titles.dropna(subset=['selftext'])  # remove NaN
    titles = titles[titles['selftext'].str.contains('removed|wiki') == False]  # remove any comments including
    # "submission removed"

    # remove punctuation
    # titles['selftext_processed'] = \
    # titles['selftext'].map(lambda x: re.sub('[\([{."\',?!})\]]', '', x))  # [\([{})\]] for regex
    # titles['selftext_processed'].map(lambda x: re.sub(r'^[0-9]{2}[fm]$', '', x))  # replace all ##m/f

    # Convert the titles to lowercase
    titles['selftext_processed'] = \
    titles['selftext'].map(lambda x: x.lower())

    # Print out the first rows of papers
    # print(titles['title_processed'].head())

    # import wordcloud library

    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'im', 'hes', 'shes', 'like', 'youre', 'youll',
                       'subreddit', 'moderator', 'https', 'relationship', 'gt'])

    # print(stop_words)

    def sent_to_words(sentences):
        for sentence in sentences:
            # deacc=True removes punctuations
            yield (gensim.utils.simple_preprocess(str(sentence),
                                                  deacc=True))

    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc))
                 if word not in stop_words] for doc in texts]

    data = titles.selftext_processed.values.tolist()
    data_words = list(sent_to_words(data))  # list()

    # remove stop words
    data_words = remove_stopwords(data_words)

    # flatten list
    from itertools import chain
    worrds = list(chain.from_iterable(data_words))
    worrrds = ' '.join(worrds)
    print(worrrds)

    # Create a WordCloud object
    wordcloud = WordCloud(
        font_path='/Users/eunjikim/PycharmProjects/rRelationships/fonts/ttf/JetBrainsMono-Regular.ttf',
        background_color="white",
        max_words=100,
        contour_width=3,
        colormap='pink',
        collocations=True)

    # Generate a word cloud
    wc = wordcloud.generate(worrrds)

    # Visualize the word cloud
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('/Users/eunjikim/PycharmProjects/rRelationships/wordcloud_comments_last.png')
    plt.show()

    # print(data_words[:1][0][:60])

    # Create Dictionary
    id2word = corpora.Dictionary(data_words)

    # Create Corpus
    texts = data_words

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # View
    # print(corpus)

    # number of topics
    num_topics = 10

    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics)

    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]


if __name__ == "__main__":
    main()
