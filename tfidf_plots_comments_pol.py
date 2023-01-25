import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels

dirstr = '/Users/eunjikim/PycharmProjects/rRelationships/tfidf_summaries/'
comments = 'tfidf_comments_summary_1.csv'
print(dirstr + comments)
# load dataset
df = pd.read_csv(dirstr + comments)  # summary file for these things anyway
df.columns = ['term_pp', '2012/05/07', '2013/07/30', '2014/06/30', '2015/01/24', '2015/05/24', '2015/09/10',
              '2015/12/15', '2016/04/27', '2016/09/02', '2017/01/18', '2017/05/04', '2017/08/13', '2017/12/20',
              '2018/05/24', '2018/09/01', '2018/12/22', '2019/04/16', '2019/09/03', '2020/01/31', '2020/07/23',
              '2021/02/24', '2021/09/22', '2022/03/28', '2022/10/01']
# df = df.drop(columns='2012/04/24')
print(df.columns)
# df.columns = pd.to_datetime(df.columns)


df_top_15 = df.head(15) # 25 terms? # just for top 100
# >2p
df_2p = df.loc[
                (df['term_pp'] == 'control') |
                (df['term_pp'] == 'trust') |
                (df['term_pp'] == 'care') |
                (df['term_pp'] == 'understand') |
                (df['term_pp'] == 'happy') |
                (df['term_pp'] == 'therapy') |
                (df['term_pp'] == 'healthy') |
                (df['term_pp'] == 'abuse') |
                (df['term_pp'] == 'single') |
                (df['term_pp'] == 'stay')
                ]

# <2%
df_less2p = df.loc[(df['term_pp'] == 'dump') |
                (df['term_pp'] == 'red flag') |
                (df['term_pp'] == 'insecurity') |
                (df['term_pp'] == 'anxiety') |
                (df['term_pp'] == 'depression') |
                (df['term_pp'] == 'immature') |
                (df['term_pp'] == 'communication') |
                (df['term_pp'] == 'perspective') |
                (df['term_pp'] == 'toxic') |
                (df['term_pp'] == 'trauma') |
                (df['term_pp'] == 'mental health') |
                (df['term_pp'] == 'unhappy')
                ]

# <.7%
df_point7 = df.loc[
                (df['term_pp'] == 'intimacy') |
                (df['term_pp'] == 'patience') |
                (df['term_pp'] == 'empathy') |
                (df['term_pp'] == 'infatuation') |
                (df['term_pp'] == 'victim') |
                (df['term_pp'] == 'manipulative') |
                (df['term_pp'] == 'unhealthy') |
                (df['term_pp'] == 'open relationship') |
                (df['term_pp'] == 'cheating') |
                (df['term_pp'] == 'trust issue') |
                (df['term_pp'] == 'gaslighting')]

# <.2%
df_point2 = df.loc[
                (df['term_pp'] == 'feelings') |
                (df['term_pp'] == 'suicide') |
                (df['term_pp'] == 'narcissist') |
                (df['term_pp'] == 'monster') |
                (df['term_pp'] == 'love language') |
                (df['term_pp'] == 'harassment') |
                (df['term_pp'] == 'lust') |
                (df['term_pp'] == 'sexual abuse') |
                (df['term_pp'] == 'emotional abuse') |
                (df['term_pp'] == 'leave relationship') |
                (df['term_pp'] == 'narcissistic')]

# < 0.05%
df_point05 = df.loc[
                (df['term_pp'] == 'unhealthy relationship') |
                (df['term_pp'] == 'emotional manipulation') |
                (df['term_pp'] == 'unresolved issue') |
                (df['term_pp'] == 'lying') |
                (df['term_pp'] == 'disloyal') |
                (df['term_pp'] == 'power dynamic') |
                (df['term_pp'] == 'boundaries') |
                (df['term_pp'] == 'misogyny')]


# Swear words in the comment section
df_swears = df.loc[(df['term_pp'] == 'shit') |
                   (df['term_pp'] == 'bitch') |
                   (df['term_pp'] == 'asshole') |
                   (df['term_pp'] == 'whore') |
                   (df['term_pp'] == 'slut') |
                   (df['term_pp'] == 'asshole')
                   ]


# turn into long form
df_top_100_lf = pd.melt(df_top_15, id_vars='term_pp')
df_top_100_lf = df_top_100_lf.rename(columns={'variable': 'era', 'value': 'tf-idf'})
df_top_100_lf['era'] = pd.to_datetime(df_top_100_lf['era'])
df_top_100_lf['era'] = pd.to_numeric(pd.to_datetime(df_top_100_lf['era']))

print(df_top_100_lf.head())
# top 100
# df_top_100_lf = df_lf.head(100)
# print(df_top_100)


g = sns.lmplot(x="era", y="tf-idf", hue="term_pp", data=df_top_100_lf,
               order=2, ci=None, scatter_kws={"s": 80}, palette="Paired")
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
# grid.add_legend()
g.fig.subplots_adjust(top=.95)
g.set(xlabel="era, from 2012 through 2022", ylabel="tf-idf calculations, % of volume per 1 million comments",
      title='Top 15 most commonly used terms in r/Relationships comments, all-time')

plt.savefig('/Users/eunjikim/PycharmProjects/rRelationships/tfidf_plots_post_comments_top15.png')
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
