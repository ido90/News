import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from time import time
from pprint import pprint
from warnings import warn
from datetime import datetime
import itertools
import inspect
from bidi import algorithm as bidi
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from urllib.request import urlopen
from bs4 import BeautifulSoup
from Analyzer.hebrew_stopwords import hebrew_stopwords
import general_utils.utils as utils
import Scrapper.ScrapperTools as st
import Analyzer.BasicAnalyzer as ba

'''
This module was mainly intended to extract meaningful words from texts.
One of its primary tasks was intended to be Hebrew stemming, which is
particularly important for Bag-of-Words based classification.

The following article suggests several NLP tools in Hebrew:
https://github.com/iddoberger/awesome-hebrew-nlp
However, only few seem to offer stemming, and none seems to
easily interface from Python.
Note that stemming in Hebrew is very tricky, and applying some simple
rules of thumb may cause more damage to the data than cleaning.

Currently no stemming is implemented in this package, which is
one of its primary limitations.
'''

word_chars_filter = '\.|,|"|\(|\)|,|;|:|\t'

def get_all_words(texts, chars_to_remove=word_chars_filter,
                  stopwords=hebrew_stopwords, filter_fun=None):
    if isinstance(texts,str): texts = [texts]
    all_words = [re.split(' | - |\t|\n\r|\n',txt)
                 for txt in texts]
    all_words = [re.sub(chars_to_remove, '', w.strip())
                 for text in all_words for w in text]
    all_words = [w for w in all_words if w and w not in stopwords]
    if filter_fun:
        all_words = [w for w in all_words if filter_fun(w)]

    return all_words

def get_vocabulary(df=None, col='text', texts=None,
                   required_freq=5, filter_fun=None):
    all_words = get_all_words(df[col] if texts is None else texts)
    tab = utils.table(all_words, -1)
    if filter_fun:
        tab = [t for t in tab if filter_fun(t[0])]
    unique_words = tuple(t[0] for t in tab if t[1]>=required_freq)
    return unique_words

def plot_words_repetitions(tab):
    f, axs = plt.subplots(1, 1)
    axs.plot(list(range(101)),
             utils.dist([t[1] for t in tab], list(range(101)))[2:],
             'k-')
    axs.set_yscale('log')
    axs.set_xlim((0,100))
    axs.set_xlabel(f'Quantile [%]\n(100% = {len(tt):d} words)', fontsize=12)
    axs.set_ylabel('Repetitions', fontsize=12)
    axs.set_title('Frequency of Words in Articles\n'+
                  '(in Hebrew without stopwords)', fontsize=14)
    utils.draw()

if __name__ == "__main__":
    df = ba.load_data(r'..\Data\articles')
    all_words = get_all_words(df.text,
                              filter_fun=lambda w: any('א'<=c<='ת' for c in w))
    tt = utils.table(all_words, -1)
    print(f'Words:\t{len(all_words):d}')
    print(f'Unique words:\t{len(tt):d}')
    print('Most repeating words:')
    pprint(tt[:10])
    plot_words_repetitions(tt)
    plt.show()
