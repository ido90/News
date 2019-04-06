import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import re
from time import time
from bidi import algorithm as bidi
from pprint import pprint
from warnings import warn
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import general_utils.utils as utils
import Scrapper.ScrapperTools as st
import Analyzer.BasicAnalyzer as ba
import Analyzer.Semantics as sm


#################   MAIN WRAPPERS   #################

def main(df, xs, ys, classifiers, **kwargs):
    assert(len(xs)==len(ys)), "Inconsistent configuration."
    fig, axs = plt.subplots(len(xs), 2)
    for i,(x,y) in enumerate(zip(xs,ys)):
        d = df.copy()
        if x!='sentence':
            d = d[~d.blocked]
        if y=='section':
            d = filter_major_sections(d)
        ax = axs if len(xs)==1 else axs[i,:]
        prepare_data_and_test(d, classifiers, x, y, (fig,ax), **kwargs)

def prepare_data_and_test(df, classifiers, x='article', y='source',
                          fig=None, t0=time(), add_heuristics=True,
                          force_balance=True, diagnosis=False):
    # convert to pairs (x,y) and split to train & test
    data = get_labeled_raw_data(df, verbose=1, force_balance=force_balance,
                                x_resolution=x, y_col=y)
    X_train, X_test, y_train, y_test = \
        train_test_split(data[0], data[1], test_size=0.2, random_state=0)
    print(f'Train & test groups defined ({time()-t0:.0f} [s]).')

    # transform x to features
    voc = sm.get_vocabulary(texts=X_train, required_freq=100,  # TODO reduce to 10?
                            filter_fun=lambda w: any('א' <= c <= 'ת' for c in w))
    print(f'Vocabulary of {len(voc):d} words is set ({time()-t0:.0f} [s]).')
    X_train = extract_features(X_train, voc, add_heuristics=add_heuristics)
    X_test = extract_features(X_test, voc, add_heuristics=add_heuristics)
    print(f'Features extracted ({time()-t0:.0f} [s]).')

    # train & test
    res = test_models(X_train, X_test, y_train, y_test, classifiers,
                      diagnosis=diagnosis, t0=t0)
    print(f'Test finished ({time()-t0:.0f} [s]).')
    if fig is None:
        fig = plt.subplots(1, 2)
    plt.figure(fig[0].number)
    plot_results(res, fig[1], x + ' -> ' + y,
                 100 / len(np.unique(y_train)))


#################   DATA PRE-PROCESSING   #################

SEPARATOR = {
    'article': 'NEVER_SPLIT',
    'paragraph': '\n|\n\r',
    'sentence': '\. |\.\n|\.\r',
    'word': ' | - |\t|\n\r|\n'
}

def filter_major_sections(df):
    df.section[df.section=='כסף'] = 'כלכלה'
    df = df[df.section.isin(('חדשות','כלכלה','כסף','ספורט','אוכל'))]
    return df

def get_labeled_raw_data(df, y_col='source', x_col='text',
                         x_resolution='article', force_balance=True,
                         min_words=10, verbose=0):
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

def extract_features(raw_texts, voc, add_heuristics=True, normalize=False):
    features = pd.DataFrame({
        **(texts_heuristics(raw_texts) if add_heuristics else {}),
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
            [np.mean(['a'<=c<='z' or 'A'<=c<='Z'
                      for w in sm.get_all_words(s) for c in w])
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


#################   TRAIN & TEST   #################

def test_models(X_train, X_test, y_train, y_test,
                models, n_buckets=5, t0=time(), diagnosis=False, verbose=2):
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
        if diagnosis:
            if models[m][0]==Perceptron:
                perceptron_diagnosis(model, list(X_train.columns))
            elif models[m][0] == DecisionTreeClassifier:
                tree_diagnosis(model, list(X_train.columns))
            elif models[m][0] == RandomForestClassifier:
                random_forest_diagnosis(model, list(X_train.columns))
            elif models[m][0] == MultinomialNB:
                naive_bayes_diagnosis(model, list(X_train.columns))
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
        ax.set_title(title + f'\n({test:s})', fontsize=14)
        ax.set_xlabel('Training samples', fontsize=12)
        ax.set_ylabel('Accuracy [%]', fontsize=12)
        ax.set_xlim((n_samples[0],n_samples[-1]))
        ax.set_ylim((0,101))
        ax.legend(loc='upper left')
    utils.draw()


#################   MODEL DIAGNOSIS   #################

def perceptron_diagnosis(model, col_names=None, fig=None):
    # input validation
    if len(model.coef_)<=2:
        raise NotImplementedError('Binary classification diagnosis is ' +
                                  'currently not supported.')
    if fig is None:
        fig = plt.subplots(1,1)
    plt.figure(fig[0].number)
    if col_names is None:
        col_names = list(range(len(model.coef_[0])))
    col_names = [bidi.get_display(nm) for nm in col_names]
    # get std of coefficients
    coef_std = [np.std(model.intercept_)] + \
               [np.std([cfs[i] for cfs in model.coef_])
                for i in range(len(model.coef_[0]))]
    # plot
    utils.barplot(fig[1], ['intercept']+col_names, coef_std, vertical_xlabs=True,
                  title=f'Perceptron Diagnosis ({model.n_iter_:d} iterations)',
                  xlab='Coefficient',
                  ylab='STD(coef) over classes\n' +
                       '(TODO note: x values are not necessarily normalized)')
    utils.draw()

def tree_diagnosis(model, col_names=None, fig=None):
    # input validation
    if fig is None:
        fig = plt.subplots(1,1)
    plt.figure(fig[0].number)
    if col_names is None:
        col_names = list(range(len(model.feature_importances_)))
    col_names = [bidi.get_display(nm) for nm in col_names]
    # plot
    utils.barplot(fig[1], col_names, model.feature_importances_, vertical_xlabs=True,
                  title=f'Decision Tree Diagnosis (Depth: {model.tree_.max_depth:d})',
                  xlab='Feature',
                  ylab='Gini importance')
    utils.draw()

def random_forest_diagnosis(model, col_names=None, fig=None):
    # input validation
    if fig is None:
        fig = plt.subplots(1,1)
    plt.figure(fig[0].number)
    if col_names is None:
        col_names = list(range(len(model.feature_importances_)))
    col_names = [bidi.get_display(nm) for nm in col_names]
    # plot
    utils.barplot(fig[1], col_names, model.feature_importances_, vertical_xlabs=True,
                  title=f'Random Forest Diagnosis ({len(model.estimators_):d} trees)',
                  xlab='Feature',
                  ylab='Average Gini importance')
    utils.draw()

def naive_bayes_diagnosis(model, col_names=None, fig=None):
    # input validation
    if fig is None:
        fig = plt.subplots(1,1)
    plt.figure(fig[0].number)
    if col_names is None:
        col_names = list(range(len(model.feature_importances)))
    col_names = [bidi.get_display(nm) for nm in col_names]
    # get std of coefficients
    log_probs_std = [np.std([lp[i] for lp in model.feature_log_prob_])
                     for i in range(len(model.feature_log_prob_[0]))]
    # plot
    utils.barplot(fig[1], col_names, log_probs_std, vertical_xlabs=True,
                  title=f'Naive Bayes Diagnosis',
                  xlab='Feature',
                  ylab='STD(log probability) over classes')
    utils.draw()


#################   MAIN   #################

if __name__ == "__main__":
    df = ba.load_data(r'..\Data\articles')

    # configuration
    xs = ['article','article','sentence']
    ys = ['source','section','section']

    classifiers = {
        'Perceptron': (Perceptron, {'max_iter': 1000, 'tol': -np.inf, 'early_stopping': True, 'validation_fraction':0.2}),
        'Tree': (DecisionTreeClassifier, {'criterion':'entropy', 'min_samples_leaf':2, 'max_depth':12}),
        'Random Forest': (RandomForestClassifier, {'criterion':'entropy', 'n_estimators':40, 'max_depth':4, 'min_samples_leaf':2, 'oob_score':True}),
        'Naive Bayes': (MultinomialNB, {'alpha':2}), # alpha = number of fictive counts in Laplace (or Lidstone) smoothing.
        'SVM': (SVC, {'kernel':'rbf', 'gamma':'scale'}),
        'Neural Network': (MLPClassifier, {'hidden_layer_sizes':(100,), 'activation':'tanh', 'learning_rate':'constant', 'max_iter':1000, 'early_stopping':True, 'validation_fraction':0.2})
    }

    # run
    main(df, xs[:3], ys[:3], classifiers, force_balance=True, diagnosis=False)
    main(df, xs[:3], ys[:3], classifiers, force_balance=True, diagnosis=False, add_heuristics=False)
    plt.show()


# note: force_balance causes loss of data (to force equal classes),
# which could be avoided by using 'class_weight':'balanced' in the classifiers,
# yet it was decided to force balance in order to make sure that
# the classification can't be won by simple classifier bias.


# TODO
# check CountVectorizer
#    http://www.insightsbot.com/blog/R8fu5/bag-of-words-algorithm-in-python-introduction
# feature selection: only BoW features!/
#                    lasso (built-in sklearn?) /
#                    according to perceptron's (normalized) betas /
#                    according to accuracy of estimation when used alone /
#                    PCA?
# features engineering?
# think about more interesting classification problems to solve
