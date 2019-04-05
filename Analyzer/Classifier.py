import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import re
from time import time
from pprint import pprint
from warnings import warn
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
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

def get_labeled_raw_data(df, y_col='source', x_col='text',
                         x_resolution='article', force_balance=True,
                         min_words=4, verbose=0):
    X0 = [x for x in df[x_col]]
    Y0 = [y for y in df[y_col]]

    # split text to desired units (e.g. sentences)
    sep = SEPARATOR[x_resolution]
    X,Y = list(), list()
    for x0,y0 in zip(X0,Y0):
        split_x = [s.strip().strip(sm.word_chars_filter) for s in
                   list(filter(None, re.split(sep, x0)))]
        split_x = [s for s in split_x if ba.count_words(s)>=min_words]
        X.extend(split_x)
        Y.extend(len(split_x) * [y0])

    # force balanced classes
    if force_balance:
        class_size = np.min([np.sum(y==y0 for y in Y)
                             for y0 in np.unique(Y)])
        for y0 in np.unique(Y):
            all_samples = [i for i,y in enumerate(Y) if y==y0]
            if verbose >= 1:
                print(f'{y0:s}:\t{class_size:d}/{len(all_samples):d} samples kept.')
            to_delete = random.sample(all_samples, len(all_samples)-class_size)
            to_delete.sort(reverse=True)
            for i in to_delete:
                del X[i]
                del Y[i]

    #return tuple((x,y) for x,y in zip(X,Y))
    return (X,Y)

def extract_features(raw_texts, voc, normalize=False):
    features = pd.DataFrame({
        **texts_heuristics(raw_texts),
        **texts_to_words(raw_texts, voc)
    })
    if normalize:
        for c in features.columns:
            features[c] = 0 if np.std(features[c]==0) else \
                (features[c]-np.mean(features[c])) / np.std(features[c])
    return features

def texts_heuristics(texts):
    return {
        'n_words': ba.count_words(texts),
        'chars_per_word': np.array([len(s) for s in texts]) /
                          np.array(ba.count_words(texts)),
        'heb_chars_rate': np.nan_to_num([np.mean(['א'<=c<='ת'
                                    for w in sm.get_all_words(s) for c in w])
                           for s in texts]),
        'eng_chars_rate': np.nan_to_num(
            [np.mean(['a'<=c<='z' or 'A'<=c<='Z' for w in sm.get_all_words(s) for c in w])
             for s in texts]),
        'num_chars_rate': np.nan_to_num([np.mean(['0'<=c<='9'
                                    for w in sm.get_all_words(s) for c in w])
                           for s in texts])
    }

# DEPRECATED: inefficient implementation (~X5 running time).
# def texts_to_words(texts, voc):
#     words_per_text = tuple(sm.get_all_words(txt) for txt in texts)
#     return {w: [ws.count(w) for ws in words_per_text] for w in voc}

def texts_to_words(texts, voc):
    sep = SEPARATOR['word']
    count = {w: len(texts)*[0] for w in voc}
    for i,txt in enumerate(texts):
        for w in re.split(sep,txt):
            w = re.sub(sm.word_chars_filter, '', w).strip()
            if w in count:
                count[w][i] += 1
    return count

def test_models(X_train, X_test, y_train, y_test,
                models, n_buckets=10, t0=time(), verbose=3):
    n_samples = [int(len(X_train)/n_buckets) * (i+1)
                 for i in range(n_buckets)]
    n_samples[-1] = len(X_train)
    res = {'train group':[n_samples,{}], 'test group':[n_samples,{}]}
    for m in models:
        res['train group'][1][m] = list()
        res['test group'][1][m] = list()
        for i,n in enumerate(n_samples):
            if verbose >= 3:
                print(f'\t{m:s}: group {i+1:d}/{len(n_samples):d} ' +
                      f'({time()-t0:.0f} [s])...')
            # train
            model = models[m][0](**models[m][1])
            model.fit(X_train[:n], y_train[:n])
            # test
            res['train group'][1][m].append(
                100*np.mean([y1==y2 for y1,y2 in
                             zip(y_train[:n],model.predict(X_train[:n]))]))
            res['test group'][1][m].append(
                100*np.mean([y1==y2 for y1,y2 in
                               zip(y_test,model.predict(X_test))]))
        if verbose >= 2:
            print(f'{m:s} finished ({time()-t0:.0f} [s]).')
    return res

def plot_results(res, axs, title='Test Classification', reference=None):
    for i,test in enumerate(res):
        ax = axs[i]
        n_samples = res[test][0]
        # plot reference
        if reference is not None:
            ax.plot((n_samples[0], n_samples[-1]),
                    2 * [reference],
                    'k:', label='Random')
        # plot actual results
        for model in res[test][1]:
            accuracy = res[test][1][model]
            ax.plot(n_samples, accuracy, label=model)
        ax.set_title(title + f' ({test:s})', fontsize=14)
        ax.set_xlabel('Training samples', fontsize=12)
        ax.set_ylabel('Accuracy [%]', fontsize=12)
        ax.set_xlim((n_samples[0],n_samples[-1]))
        ax.set_ylim((0,101))
        ax.legend()
    utils.draw()


if __name__ == "__main__":
    t0 = time()
    df = ba.load_data(r'..\Data\articles')
    print(f'Data loaded ({time()-t0:.0f} [s]).')
    data = get_labeled_raw_data(df[~df.blocked], verbose=1,
                                x_resolution='article', y_col='source')
    print(f'Raw (x,y) data created ({time()-t0:.0f} [s]).')

    # TODO: choosing the vocabulary by frequency of words in
    #       both train & test data sets is a slight cheat.
    voc = sm.get_vocabulary(df, required_freq=300, # TODO reduce to 10?
                            filter_fun=lambda w: any('א'<=c<='ת' for c in w))
    print(f'Vocabulary of {len(voc):d} words is set ({time()-t0:.0f} [s]).')

    X = extract_features(data[0], voc, normalize=True)
    print(f'Features extracted ({time()-t0:.0f} [s]).')
    X_train, X_test, y_train, y_test =\
        train_test_split(X, data[1], test_size=0.2, random_state=0)
    print(f'Train & test groups defined ({time()-t0:.0f} [s]).')

    res = test_models(X_train, X_test, y_train, y_test,
                      {'Perceptron':(Perceptron,{'max_iter':3000}),
                       'Tree':(DecisionTreeClassifier,{})},
                      t0=t0)
    print(f'Test finished ({time()-t0:.0f} [s]).')
    fig,axs = plt.subplots(2,2)
    plot_results(res, axs[0,:], 'Classification: article -> source',
                 100/len(np.unique(y_train)))
    # TODO models, models configurations, and more stats (e.g. train error, model main features)
    # TODO choose more problems rather than sentence->source
    plt.show()

# stem
# choose models
# engineer features / feature selection?
# think about experiments
