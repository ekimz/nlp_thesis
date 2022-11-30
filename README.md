# Thesis: Natural Language Processing on subreddit r/Relationships

# About this repository
This repository includes all files that were used in the analysis of my thesis for the QMSS MA program; each file in the repo is described below as a part of the process. This is a work in progress, so not all techniques mentioned above may have already been used.

This project is to be completed by December 31, 2022.

# Project Premise
With the rise of divorce rates and increase in visibility of toxic traits in relationships alongside the cultural wave of feminism in the United States, the way people view romantic relationships seems to have changed over time. This project analyses the subreddit r/Relationships using Natural Language Processing (NLP) techniques such as sentiment analysis, tf-idf, LDA, and word clouds as well as a few time series visualizations to see if there have actually been a shift in vocabulary used by commenters to provide different types of advice in the past decade.

# The Dataset
The dataset used was retrieved using Pushshift.io, so upvotes and awards information was not retrieved as gathering textual data over a longer period of time prioritized. This means that posts and comments that were popular are not given priority, although a multitude of comments may refer to the same post. Although comments can be by one person through a long chain, each comment is counted once. The pros and cons of this will be described in the thesis.


# 1. Data Extraction & Cleaning

File(s): pd2csv_posts.py & pd2csv_comments.py
Modified code from @Watchful1 to pull posts and comments into organized dataframes into a .csv file instead of .txt file
Reference: https://github.com/Watchful1/Sketchpad/blob/master/postDownloader.py

File(s): csv_reads.py
Used answer by Benjamin Ziepert to split csv of posts & comments into 100k and 1m lines each
Reference: https://stackoverflow.com/questions/36445193/splitting-one-csv-into-multiple-files

File(s): split_csv_deleteds
Separated removed posts/comments as its own dataset to perform analysis on posts/comments that users left posted for better analysis as any removed or deleted posts/comments can skew results


# 2. Sentiment Analysis

File(s): sentiment_scoring.py
Performed sentiment analysis on each post/comment of every csv file
Reference: negative-words.txt & positive-words.txt

File(s): sentiment_score_analysis.py
Summarized sentiment analysis scores from each csv file to show min, max, mean, median, std, & var in one file for visualization; used on comments & posts


# 3. TF-IDF

File(s): tf_idf_analysis.py
Currently working on performing tf-idf on each section for topic analysis


# the rest of the plan:
- Topic analysis by sentiment score
- Word frequencies per "era" (word clouds, tf-idf)
- LDA / n-gram / bag of words on each set
- Word Clouds per tf-idf batch # comment
- Time series analysis 
