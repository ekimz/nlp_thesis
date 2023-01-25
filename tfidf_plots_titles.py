import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels

dirstr = '/Users/eunjikim/PycharmProjects/rRelationships/tfidf_summaries/'
# comments = '.csv'
post_titles = 'tfidf_titles_summary.csv'
print(dirstr + post_titles)
# load dataset
df = pd.read_csv(dirstr + post_titles)  # summary file for these things anyway
df.columns = ['term_pp', '2012/04/24', '2015/03/30', '2015/10/28', '2016/04/21', '2016/09/27', '2017/03/01',
              '2017/07/30', '2018/02/04', '2018/07/20', '2018/12/21', '2019/05/25', '2019/11/12', '2020/04/30',
              '2020/11/19', '2021/05/19', '2021/11/07', '2022/04/23', '2022/09/30']
df = df.drop(columns='2012/04/24')
print(df.columns)
# df.columns = pd.to_datetime(df.columns)

df_gay = df.loc[(df['term_pp'] == 'gay') |
                (df['term_pp'] == 'sexual orientation') |
                (df['term_pp'] == 'disown') |
                (df['term_pp'] == 'queer abuse') |
                (df['term_pp'] == 'grooming') |
                (df['term_pp'] == 'queer') |
                (df['term_pp'] == 'faggot') |
                (df['term_pp'] == 'came out') |
                (df['term_pp'] == 'closet') |
                (df['term_pp'] == 'lesbian')|
                (df['term_pp'] == 'bisexual') |
                (df['term_pp'] == 'pansexual') |
                (df['term_pp'] == 'transgender')]

df_religion = df.loc[
                (df['term_pp'] == 'disown') |
                (df['term_pp'] == 'grooming') |
                (df['term_pp'] == 'religious') |
                (df['term_pp'] == 'conservative') |
                (df['term_pp'] == 'god') |
                (df['term_pp'] == 'jesus')]

df_topics = df.loc[
                (df['term_pp'] == 'emotional manipulation') |
                (df['term_pp'] == 'abuse') |
                (df['term_pp'] == 'break up') |
                (df['term_pp'] == 'cheating') |
                (df['term_pp'] == 'trust issue') |
                (df['term_pp'] == 'red flag') |
                (df['term_pp'] == 'gaslighting') |
                (df['term_pp'] == 'trust') |
                (df['term_pp'] == 'narcissist')]

df_whomst = df.loc[
                (df['term_pp'] == 'girlfriend') |
                (df['term_pp'] == 'boyfriend') |
                (df['term_pp'] == 'ex') |
                (df['term_pp'] == 'husband') |
                (df['term_pp'] == 'wife') |
                (df['term_pp'] == 'partner') |
                (df['term_pp'] == 'fiance') |
                (df['term_pp'] == 'fwb') |
                (df['term_pp'] == 'best friend') |
                (df['term_pp'] == 'coworker') |
                (df['term_pp'] == 'mother') |
                (df['term_pp'] == 'father') |
                (df['term_pp'] == 'brother') |
                (df['term_pp'] == 'sister') |
                (df['term_pp'] == 'sibling') |
                (df['term_pp'] == 'cousin')]


df_top_15 = df.head(15) # 25 terms?

# turn into long form
df_top_100_lf = pd.melt(df_whomst, id_vars='term_pp')
df_top_100_lf = df_top_100_lf.rename(columns={'variable': 'era', 'value': 'tf-idf'})
df_top_100_lf['era'] = pd.to_datetime(df_top_100_lf['era'])
df_top_100_lf['era'] = pd.to_numeric(pd.to_datetime(df_top_100_lf['era']))


print(df_top_100_lf.head())
# top 100
# df_top_100_lf = df_lf.head(100)
#print(df_top_100)


g = sns.lmplot(x="era", y="tf-idf", hue="term_pp", data=df_top_100_lf,
               order=2, ci=None, scatter_kws={"s": 80})
#
# grid = sns.FacetGrid(df_top_100_lf,
#                      col="term_pp",
#                      hue="term_pp",
#                      col_wrap=5)
#
# print(grid)
# grid.map(sns.regplot,
#          x="era",
#          y="tf-idf")

g.add_legend()
g.fig.subplots_adjust(top=.95)
g.set(xlabel="era, from 2012 through 2022", ylabel="tf-idf calculations, % of volume per 100,000 posts",
      title='Relations featured in r/Relationships post titles, all-time')

# grid.add_legend()


plt.savefig('/Users/eunjikim/PycharmProjects/rRelationships/tfidf_plots_post_titles_whomsts.png')
plt.show()
print('saved image!')

# # create scatter plot
# print("Scatter Plot:  ")
# plt.scatter(df["X"], df["Y"])
# plt.show()
#
# # plot trend line for terms
# z = np.polyfit(x, y, 1)
# p = np.poly1d(z)
# plt.plot(x,p(x),"r--")
#
# plt.show()

#  y-axis : terms
#  Xaxis: dates

