# Thesis: Natural Language Processing on subreddit r/Relationships

### About this repository
This repository includes all files that were used in the analysis of my thesis for the QMSS MA program at Columbia University; each file in the repo is described below as a part of the process. There are some extraneous files that were referenced in the repo; those that were referenced and used are listed below, others that were rendered irrelevant are not included.

Each CSV file for posts range between 150~450MB and for comments range 600~950MB, with 18 and 24 files respectively; if I can figure out a way to put it on GitHub that isn’t absurd I will include it as reference material.

Although I was able to create visualizations from the data that indicated that my hypothesis was correct, I also would like to provide statistical analyses regarding those visualizations. As the plots were drawn in Seaborn, trendlines need to be calculated separately as those functions cannot be pulled so I am currently working on providing coherence scores and t-statistics. Thus I suppose this is technically still a work in progress; however, the thesis itself has been turned in and graded so any additional changes made will be demarcated as such.

This project was completed on December 30, 2022.

### Project Premise
With the rise of divorce rates and increase in visibility of toxic traits in relationships alongside the cultural wave of feminism in the United States, the way people view romantic relationships seems to have changed over time. This project analyses the subreddit r/Relationships using Natural Language Processing (NLP) techniques such as sentiment analysis, tf-idf, LDA, and word clouds as well as a few time series visualizations to see if there have actually been a shift in vocabulary used by commenters to provide different types of advice in the past decade.

### The Dataset
The dataset used was retrieved using Pushshift.io, so upvotes and awards information was not retrieved as gathering textual data over a longer period of time prioritized. This means that posts and comments that were popular are not given priority, although a multitude of comments may refer to the same post. Although comments can be by one person through a long chain, each comment is counted separately as its own comment, considering that the statistical significance of one person posting comments frequently most likely would not affect the analysis, as there are approximately 24 million comments, and the ability to have commented enough to affect that volume of comments within 14 years honestly seems mildly aggressive (to be at 1% of the dataset, a user would have needed to have posted 46 times per day every day for 14 years, from 2008 through 2022). The pros and cons of this are described in the thesis.

### Some Tools
- PyCharm - to code
- Gigasheet - to read the CSVs
- Google - to figure out why I was erroring out

#Methodology

All original files are in this repo; each file described below will additionally be provided as skeletons that should replicate results with any subreddit. Results will be in PDF format for dataset.

## 1. Data Extraction & Cleaning

File(s): pd2csv_posts.py & pd2csv_comments.py  
Modified code from @Watchful1 to pull posts and comments into organized dataframes into a .csv file instead of .txt file. Includes the following columns:

Posts:
- ‘subreddit'
- ‘title'
- ‘created_utc'
- ‘selftext'
- ‘upvote_ratio'
- ‘score'
- ‘permalink'
- ‘id'

Comments:
- 'subreddit'
- ‘created_utc’
- 'selftext'
- 'score'
- 'permalink'
- ‘id'

Reference: https://github.com/Watchful1/Sketchpad/blob/master/postDownloader.py

File(s): csv_reads.py
Split large CSV of posts & comments into 100k and 1m line CSVs each
Reference: https://stackoverflow.com/questions/36445193/splitting-one-csv-into-multiple-files, modified answer by Benjamin Ziepert

File(s): split_csv_deleteds  
Separated removed posts/comments as its own dataset (CSV) to perform analysis on posts/comments that users left posted for better analysis as any deleted posts/comments can skew results, as all deleted comments only have ‘removed’ as the content for text data without the text that existed prior.


## 2. Sentiment Analysis

File(s): sentiment_scoring.py (on posts), sentiment_scoring_comments.py
Performed sentiment analysis on each post/comment of every CSV file, in a for-loop for files that are all in one folder
Reference: negative-words.txt & positive-words.txt

File(s): sentiment_score_analysis.py (for comments), sentiment_score_analysis_sandbox.py (used for posts)
Summarized sentiment analysis scores from each CSV file to show min, max, mean, median, std, & var in one file for visualization; used on comments & posts  


## 3. Latent Dirichlet Allocation (LDA) and Word Clouds

File(s): lda_model_comments.py, lda_model_posts.py
Cleaned provided post or comment file for consistency, dropped unnecessary columns, duplicate posts/comments, NA; for comments, dropped any comments containing ‘removed’ or ‘wiki’ as those pertained to comments created by bots redirecting users to the rules. Made all lowercase and extended stop word list. Created Word Cloud using cleaned data from input file (in my case, the earliest file and most recent file). Built LDA model with 10 topics from file. Entire file prints one Word Cloud and ten LDA topics per input file.

Reference:
This helped with the LDA portion
https://github.com/marcmuon/nlp_yelp_review_unsupervised/blob/master/notebooks/2-train_corpus_prep_and_LDA_train.ipynb


## 4. tf-idf time series plots

File(s): tf_idf_analysis.py, tf_idf_analysis_sandbox.py
Compute tf-idf frequencies for each file to obtain ratio of the total number of times a word was used to the total number of words in each file.


File(s): tfidf_loop.py
Created a loop to preprocess data of each CSV file, calculate tf-idf calculations and save as its own file.


Files(s): merge_em.py
Merge files from tfidf_loop.py into one CSV file, with each row representing a unigram or bigram, and each column being the era of the file; this was done for tf-idf calculated for all comments, and for posts, was done separately for post titles and post content.


File(s): tfidf_plots_comments_pol.py, tfidf_plots_comments.py, tfidf_plots_titles.py
Converted dates to number and plotted time series scatterplots of relevant terms (unigrams and bigrams). Organized words to plot by percentage groups for cleaner visualizations (2%, <2%, <0.7%, <0.2%, <0.05%).


Currently working on calculating significant results from tf-idf portion
