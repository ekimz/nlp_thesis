#!/usr/bin/env python
"""
Minimal Example
===============
Generating a square wordcloud from the US constitution using default arguments.
"""

# Loop through batches
def to_txt_batches(src_csv, dst_dir, size=100000, index=False):
    import pandas as pd
    import math

    # Read source csv
    df = pd.read_csv(src_csv)

    for i in range(math.ceil(len(df) / size)):

        fname = dst_dir + '/Batch_' + str(i + 1) + '.csv'
        df[low:high].to_csv(fname, index=index)

        # Update selection
        low = high
        if (high + size < len(df)):
            high = high + size
        else:
            high = len(df)





import os

from os import path
from wordcloud import WordCloud

# get data directory (using getcwd() is needed to support running example in generated IPython notebook)
d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

# Read the whole text.
text = open(path.join(d, 'constitution.txt')).read()

# Generate a word cloud image
wordcloud = WordCloud().generate(text)

# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

# lower max_font_size
wordcloud = WordCloud(max_font_size=40).generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# The pil way (if you don't have matplotlib)
# image = wordcloud.to_image()
# image.show()
