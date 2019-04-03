import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
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
import general_utils.utils as utils
import Scrapper.ScrapperTools as st
import Analyzer.BasicAnalyzer as ba



if __name__ == "__main__":
    df = ba.load_data(r'..\Data\articles')
    all_words = [list(filter(None,re.split(' | - |\t|\n\r|\n',txt)))
                 for txt in df.text]
    all_words = [w for text in all_words for w in text]
    tt = utils.table(all_words, -1)
    print(len(all_words))
    print(len(tt))
    print(tt[:10])
    print(utils.dist([t[1] for t in tt], (0,50,70,80,85,90,95,97,99,100)))

# TODO Hebrew stemmer
# https://www.google.com/url?q=https%3A%2F%2Fgithub.com%2Fiddoberger%2Fawesome-hebrew-nlp&sa=D&usd=2&usg=AFQjCNHD9kq16TroW54hKc-H_iY6Y8xyMA
# TODO Hebrew stopwords
# TODO look for generic tools/concepts/methods in classification and characterization of articles/texts
# TODO try some simple sklearn classifications
