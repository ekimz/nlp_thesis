import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

dirstr = '/Users/eunjikim/PycharmProjects/rRelationships/tfidf_summaries/'
comments = 'tfidf_comments_summary_1.csv'
print(dirstr + comments)

# load dataset
df = pd.read_csv(dirstr + comments)  # summary file for these things anyway
print(df.columns)
df.columns = ['term_pp', '2012/05/07', '2013/07/30', '2014/06/30', '2015/01/24', '2015/05/24', '2015/09/10',
              '2015/12/15', '2016/04/27', '2016/09/02', '2017/01/18', '2017/05/04', '2017/08/13', '2017/12/20',
              '2018/05/24', '2018/09/01', '2018/12/22', '2019/04/16', '2019/09/03', '2020/01/31', '2020/07/23',
              '2021/02/24', '2021/09/22', '2022/03/28', '2022/10/01']
print(df.columns)
# df.columns = pd.to_datetime(df.columns)

print(type(df.columns))

df_top_100 = df.head(10) # 25 terms?

# turn into long form
df_top_100_lf = pd.melt(df_top_100, id_vars='term_pp')
df_top_100_lf = df_top_100_lf.rename(columns={'variable': 'era', 'value': 'tf-idf'})
df_top_100_lf['era'] = pd.to_datetime(df_top_100_lf['era'])
df_top_100_lf['era'] = pd.to_numeric(pd.to_datetime(df_top_100_lf['era']))


print(df_top_100_lf.head())
# top 100
# df_top_100_lf = df_lf.head(100)
#print(df_top_100)


g = sns.lmplot(x="era", y="tf-idf", hue="term_pp", data=df_top_100_lf)
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


plt.savefig('/Users/eunjikim/PycharmProjects/rRelationships/tfidf_plots_comments_10.png')
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

