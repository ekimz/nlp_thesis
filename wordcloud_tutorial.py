import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import gensim
import gensim.corpora as corpora
import pyLDAvis.gensim_models
import pickle
import pyLDAvis

from wordcloud import WordCloud
from spacy.lang.en import English
from collections import Counter
from nltk.tokenize import word_tokenize
from gensim.utils import simple_preprocess
from pprint import pprint
from multiprocessing import Process
from multiprocessing import set_start_method
from gensim.models.coherencemodel import CoherenceModel

# import ssl


# try:
#    _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
# nltk.download('stopwords')
# nltk.download('punkt')

# function executed in a new process
def task():
    print('Hello from a child process', flush=True)


# set the start method
set_start_method('fork')
# create and configure a new process
process = Process(target=task)
# start the new process
process.start()
# wait for the new process to finish
process.join()

# sns.set_style("whitegrid")
# plt.rcParams['figure.dpi'] = 180
# nlp = English()
# nlp.max_length = 5000000

# import the csv file into a Pandas dataframe
posts_df = pd.read_csv("/Users/eunjikim/PycharmProjects/rRelationships/batched_posts_100k/posts_20080723-20120424.csv")

# view the shape of the data (the number of rows and columns)
# print(f"The shape of the data is: {posts_df.shape}")

# remove unnecessary columns
posts_df = posts_df.drop(columns=['Unnamed: 0', 'subreddit', 'selftext',
                                  'upvote_ratio', 'score', 'permalink', 'id'])

# Remove punctuation
posts_df['title_processed'] = \
    posts_df['title'].map(lambda x: re.sub('[,\.!?]', '', x))

# Convert all to lowercase
posts_df['title_processed'] = \
    posts_df['title_processed'].map(lambda x: x.lower())

# print out first rows of posts_df
# print(posts_df['title_processed'].head())

# Join the different processed titles together.
long_string = ','.join(list(posts_df['title_processed'].values))

# Create a WordCloud object
wordcloud = WordCloud(background_color="white",
                      max_words=5000,
                      contour_width=3,
                      contour_color='steelblue')

# Generate a word cloud
wordcloud.generate(long_string)

# Visualize the word cloud
wordcloud.to_image()

# remove stop words from corpus
stop_words = nltk.corpus.stopwords.words('english')
stop_words.extend(['n\'t', '\'s'])
keeps = ['her', 'hers', 'him', 'his', 'my', 'i']
my_stopwords = [el for el in stop_words if el not in keeps]


def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))


def remove_stopwords(words):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in words]


data = posts_df.title_processed.values.tolist()
data_words = list(sent_to_words(data))

# remove stop words
data_words = remove_stopwords(data_words)

# Create Dictionary
id2word = corpora.Dictionary(data_words)

# Create Corpus
texts = data_words

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1][0][:30])

# number of topics
num_topics = 10

# Build LDA model
lda_model = gensim.models.LdaModel(corpus=corpus,
                                   id2word=id2word,
                                   num_topics=num_topics,
                                   random_state=0,
                                   chunksize=100,
                                   alpha='auto',
                                   per_word_topics=True)

# Print keyword in the ten topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model,
                                     texts=data_words,
                                     dictionary=id2word,
                                     coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()

print('\\nCoherence Score: ', coherence_lda)

# Visualize the topics
pyLDAvis.enable_notebook()

LDAvis_data_filepath = os.path.join('./results/ldavis_prepared_' + str(num_topics))

# this is a bit time consuming - make the if statement True
# if you want to execute visualization prep yourself
if 1 == 1:
    LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)

# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)

pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_prepared_' + str(num_topics) + '.html')

LDAvis_prepared
