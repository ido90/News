import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from time import time
from pprint import pprint
from warnings import warn
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
import general_utils.utils as utils
import Scrapper.ScrapperTools as st
import Analyzer.BasicAnalyzer as ba
import Analyzer.Semantics as sm

SEPARATOR = {
    'article': 'NEVER_SPLIT',
    'paragraph': '\n|\n\r',
    'sentence': '\. |\.\n|\.\r',
    'word': ' | - |\t|\n\r|\n'
}

def get_labeled_raw_data(df, y_col='source', x_col='text', x_resolution='article'):
    X0 = [x for x in df[x_col]]
    Y0 = [y for y in df[y_col]]

    sep = SEPARATOR[x_resolution]
    X,Y = list(), list()
    for x0,y0 in zip(X0,Y0):
        split_x = [s.strip(' |.|\t|\n|\r') for s in
                   list(filter(None, re.split(sep, x0)))]
        split_x = [s for s in split_x if s]
        X.extend(split_x)
        Y.extend(len(split_x) * [y0])

    return tuple((x,y) for x,y in zip(X,Y))

def extract_features(raw_texts, voc): # TODO optimize somehow - too slow
    return pd.DataFrame({
        **texts_heuristics(raw_texts),
        **texts_to_words(raw_texts, voc)
    })

def texts_heuristics(texts):
    return {
        'n_words': ba.count_words(texts),
        'chars_per_word': np.array([len(s) for s in texts]) /
                          np.array(ba.count_words(texts)),
        'heb_chars_rate': [np.mean(['א'<=c<='ת'
                                    for w in sm.get_all_words(s) for c in w])
                           for s in texts],
        'eng_chars_rate': [np.mean(['a'<=c<='z' or 'A'<=c<='Z'
                                    for w in sm.get_all_words(s) for c in w])
                           for s in texts],
        'num_chars_rate': [np.mean(['0'<=c<='9'
                                    for w in sm.get_all_words(s) for c in w])
                           for s in texts]
    }

def texts_to_words(texts, voc):
    words_per_text = tuple(sm.get_all_words(txt) for txt in texts)
    return {w: [ws.count(w) for ws in words_per_text] for w in voc}

def test_models(X_train, X_test, y_train, y_test, models, n_buckets=10):
    n_samples = [int(len(X_train)/n_buckets) * (i+1) for i in range(n_buckets)]
    n_samples[-1] = len(X_train)
    f, axs = plt.subplots(1, 1)
    res = {}
    for m in models:
        res[m] = list()
        for n in n_samples:
            model = models[m]()
            model.fit(X_train[0:n], y_train[:n])
            yp = model.predict(X_test)
            res[m].append(100*np.mean([y1==y2 for y1,y2 in zip(yp,y_test)]))
        axs.plot(n_samples, res[m], label=m)
    # TODO plot nicer...
    axs.set_xlabel('Number of training samples', fontsize=12)
    axs.set_ylabel('Accuracy [%]', fontsize=12)
    axs.set_ylim((0,100))
    axs.legend()
    utils.draw()

if __name__ == "__main__":
    t0 = time()
    df = ba.load_data(r'..\Data\articles')
    print(f'Data loaded ({time()-t0:.0f} [s]).')
    data = get_labeled_raw_data(df, x_resolution='sentence')
    data=data[:10000] # TODO optimize feature extraction and remove this
    print(f'Raw (x,y) data created ({time()-t0:.0f} [s]).')
    voc = sm.get_vocabulary(df, required_freq=100, # TODO reduce to 10 and deal with many features
                            filter_fun=lambda w: any('א'<=c<='ת' for c in w))

    print(f'Vocabulary of {len(voc):d} words is set ({time()-t0:.0f} [s]).')
    X = extract_features([d[0] for d in data], voc) # TODO stem and filter stopwords
    print(f'Features extracted ({time()-t0:.0f} [s]).')
    y = [d[1] for d in data]
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.2, random_state=0)
    print(f'Train & test groups defined ({time()-t0:.0f} [s]).')

    test_models(X_train, X_test, y_train, y_test, {'Perceptron':Perceptron})
    # TODO models, models configurations, and more stats (e.g. train error, model main features)
    # TODO choose more problems rather than sentence->source
    plt.show()
