import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import nltk
import re

from wordcloud import WordCloud, STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')

from string import punctuation
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import LinearSVC


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# training dataset
# train = pd.read_csv('/kaggle/input/stress-analysis-in-social-media/dreaddit-train.csv')
# train.shape

# testing dataset
# test = pd.read_csv('/kaggle/input/stress-analysis-in-social-media/dreaddit-test.csv')
# test.shape

# full = pd.concat((train, test), sort=False).reset_index(drop=True)

# df = full[['text', 'subreddit']].copy()
# df.head()

# pd.DataFrame(df.subreddit.unique()).values()

# Word Clouds

plt.figure(figsize=(60, 35))

subset = df[df['subreddit'] == 'ptsd']
text = subset.text.values
cloud1 = WordCloud(stopwords=STOPWORDS,
                   background_color='pink',
                   colormap="Dark2",
                   collocations=False,
                   width=2500,
                   height=1800).generate(" ".join(text))
plt.subplot(5, 2, 1)
plt.axis('off')
plt.title("PTSD", fontsize=40)
plt.imshow(cloud1)

subset = df[df['subreddit'] == 'assistance']
text = subset.text.values
cloud2 = WordCloud(stopwords=STOPWORDS,
                   background_color='pink',
                   colormap="Dark2",
                   collocations=False,
                   width=2500,
                   height=1800
                   ).generate(" ".join(text))
plt.subplot(5, 2, 2)
plt.axis('off')
plt.title("Assistance", fontsize=40)
plt.imshow(cloud2)

subset = df[df['subreddit'] == 'relationships']
text = subset.text.values
cloud3 = WordCloud(stopwords=STOPWORDS,
                   background_color='pink',
                   colormap="Dark2",
                   collocations=False,
                   width=2500,
                   height=1800
                   ).generate(" ".join(text))
plt.subplot(5, 2, 3)
plt.axis('off')
plt.title("Relationships", fontsize=40)
plt.imshow(cloud3)

subset = df[df['subreddit'] == 'survivorsofabuse']
text = subset.text.values
cloud4 = WordCloud(stopwords=STOPWORDS,
                   background_color='pink',
                   colormap="Dark2",
                   collocations=False,
                   width=2500,
                   height=1800
                   ).generate(" ".join(text))
plt.subplot(5, 2, 4)
plt.axis('off')
plt.title("Survivors of abuse", fontsize=40)
plt.imshow(cloud4)

subset = df[df['subreddit'] == 'domesticviolence']
text = subset.text.values
cloud5 = WordCloud(stopwords=STOPWORDS,
                   background_color='pink',
                   colormap="Dark2",
                   collocations=False,
                   width=2500,
                   height=1800
                   ).generate(" ".join(text))
plt.subplot(5, 2, 5)
plt.axis('off')
plt.title("Domestic violence", fontsize=40)
plt.imshow(cloud5)

subset = df[df['subreddit'] == 'anxiety']
text = subset.text.values
cloud6 = WordCloud(stopwords=STOPWORDS,
                   background_color='pink',
                   colormap="Dark2",
                   collocations=False,
                   width=2500,
                   height=1800
                   ).generate(" ".join(text))
plt.subplot(5, 2, 6)
plt.axis('off')
plt.title("Anxiety", fontsize=40)
plt.imshow(cloud6)

subset = df[df['subreddit'] == 'homeless']
text = subset.text.values
cloud7 = WordCloud(stopwords=STOPWORDS,
                   background_color='pink',
                   colormap="Dark2",
                   collocations=False,
                   width=2500,
                   height=1800
                   ).generate(" ".join(text))
plt.subplot(5, 2, 7)
plt.axis('off')
plt.title("Homeless", fontsize=40)
plt.imshow(cloud7)

subset = df[df['subreddit'] == 'stress']
text = subset.text.values
cloud8 = WordCloud(stopwords=STOPWORDS,
                   background_color='pink',
                   colormap="Dark2",
                   collocations=False,
                   width=2500,
                   height=1800
                   ).generate(" ".join(text))
plt.subplot(5, 2, 8)
plt.axis('off')
plt.title("Stress", fontsize=40)
plt.imshow(cloud8)

subset = df[df['subreddit'] == 'almosthomeless']
text = subset.text.values
cloud9 = WordCloud(stopwords=STOPWORDS,
                   background_color='pink',
                   colormap="Dark2",
                   collocations=False,
                   width=2500,
                   height=1800
                   ).generate(" ".join(text))
plt.subplot(5, 2, 9)
plt.axis('off')
plt.title("Almost homeless", fontsize=40)
plt.imshow(cloud9)

subset = df[df['subreddit'] == 'food_pantry']
text = subset.text.values
cloud10 = WordCloud(stopwords=STOPWORDS,
                    background_color='pink',
                    colormap="Dark2",
                    collocations=False,
                    width=2500,
                    height=1800
                    ).generate(" ".join(text))
plt.subplot(5, 2, 10)
plt.axis('off')
plt.title("Food pantry", fontsize=40)
plt.imshow(cloud10)

# Creating dictionaries
df['subreddit_id'] = df['subreddit'].factorize()[0]
subreddit_id_df = df[['subreddit', 'subreddit_id']].drop_duplicates()

subreddit_to_id = dict(subreddit_id_df.values)
id_to_subreddit = dict(subreddit_id_df[['subreddit_id', 'subreddit']].values)

df.head()

stop = set(stopwords.words('english'))


# Text preprocessing
def lower(text):
    return text.lower()


def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stop])


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', punctuation))


def clean_text(text):
    text = lower(text)
    text = remove_stopwords(text)
    text = remove_punctuation(text)
    return text


# Apply function on column
df['clean_text'] = df['text'].apply(clean_text)

# Removing common words
cnt = Counter()

for text in df['clean_text'].values:
    for word in text.split():
        cnt[word] += 1

cnt.most_common(10)

freq_words = set([w for (w, wc) in cnt.most_common(10)])


def remove_freq_words(text):
    return " ".join([word for word in str(text).split() if word not in freq_words])


df["clean_text"] = df["clean_text"].apply(lambda text: remove_freq_words(text))

# # Words lemmatization
lemmatizer = WordNetLemmatizer()


def lemmatized_words(text):
    return " ".join([lematizer.lemmatize(word) for word in text.split()])


df['clean_text'] = df['clean_text'].apply(lambda text: lemmatized_words(text))

df.head()

# tf_idf
tf_idf = TfidfVectorizer(sublinear_tf=True, min_df=5, ngram_range=(1, 2), stop_words='english')

features = tf_idf.fit_transform(df.clean_text).toarray()

labels = df.subreddit_id

print("Each of the %d text is represented by %d features (TF-IDF score of unigrams and bigrams)" % (features.shape))

# Finding the three most correlated terms with each of the categories
N = 3
for subreddit, subreddit_id in sorted(subreddit_to_id.items()):
    features_chi2 = chi2(features, labels == subreddit_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("\n==> %s:" % (subreddit))
    print("  * Most Correlated Unigrams are: %s" % (', '.join(unigrams[-N:])))
    print("  * Most Correlated Bigrams are: %s" % (', '.join(bigrams[-N:])))

X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                    test_size=0.25,
                                                    random_state=20)

model = LinearSVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

metrics.accuracy_score(y_test, y_pred)
