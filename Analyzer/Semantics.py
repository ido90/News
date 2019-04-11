import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import re
import pickle
from time import time
from pprint import pprint
from warnings import warn
from datetime import datetime
import itertools
from bidi import algorithm as bidi
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from Analyzer.hebrew_stopwords import hebrew_stopwords
import general_utils.utils as utils
import Scrapper.ScrapperTools as st
import Analyzer.BasicAnalyzer as ba

'''
This module is intended to extract meaningful words from texts.

One of the primary tasks was intended to be Hebrew stemming, which is
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

SEPARATOR = {
    'article': 'NEVER_SPLIT',
    'paragraph': '\n|\n\r',
    'sentence': '\. |\.\n|\.\r',
    'word': ' | - |\t|\n\r|\n'
}

word_chars_filter = '\.|,|"|\(|\)|;|:|\t'

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

def get_all_sentences(texts):
    sep = SEPARATOR['sentence']
    sents = list()
    for txt in texts:
        sents.extend([s.strip().strip(word_chars_filter) for s in
                     list(filter(None, re.split(sep, txt)))])
    return sents

def get_vocabulary(df=None, col='text', texts=None,
                   required_freq=5, filter_fun=None, **kwargs):
    all_words = get_all_words(df[col] if texts is None else texts, **kwargs)
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

def count_parties(
        ax, df, col='text', by='source', binary_per_text=False, logscale=False,
        keys=('ליכוד', ('ביבי','נתניהו'), ('כחול לבן','כחול-לבן'), 'גנץ', 'העבודה', 'גבאי',
              ('חד"ש','תע"ל'), 'עודה', 'יהדות התורה', 'ליצמן', 'איחוד הימין', "סמוטריץ'",
              'הימין החדש','בנט', 'זהות', 'פייגלין', 'מרצ', 'זנדברג', 'ש"ס', 'דרעי',
              'כולנו', 'כחלון', ('בל"ד','רע"ם'), 'עבאס',
              ('ישראל ביתנו','ישראל-ביתנו'), 'ליברמן', 'גשר', 'אורלי לוי')
                  ):
    groups = np.unique(df[by])
    sep = SEPARATOR['word']

    count = {grp: len(keys)*[0] for grp in groups}
    for grp in groups:
        for i,txt in enumerate(df[df[by]==grp][col]):
            for j, key in enumerate(keys):
                # multi-word keys
                appears = 0
                if isinstance(key, tuple):
                    for k in key:
                        if ' ' in k:
                            appears = txt.count(k)
                            count[grp][j] += bool(appears) if binary_per_text else appears
                            if binary_per_text: break
                else:
                    k = key
                    if ' ' in k:
                        appears = txt.count(k)
                        count[grp][j] += bool(appears) if binary_per_text else appears
                if binary_per_text and appears:
                    continue
                # one-word keys
                for w in re.split(sep, txt):
                    w = re.sub('\.|,|\(|\)|;|:|\t', '', w).strip()
                    if w.endswith(key):
                        count[grp][j] += 1
                        if binary_per_text: break

    keys = tuple(k[0]+' /\n'+k[1] if isinstance(k,tuple) else k for k in keys)
    keys = tuple(bidi.get_display(k) for k in keys)
    colors = utils.DEF_COLORS
    bottom = np.array([0 for _ in keys])

    ylab = ('Texts with the expression' if binary_per_text else 'Total appearances') +\
           '\n(as end of a word)'
    for i,group in enumerate(groups):
        utils.barplot(ax, keys, count[group], bottom=bottom, plot_bottom=False,
                      ylab=ylab, title='Frequency of appearance', vertical_xlabs=True,
                      colors=colors[i%len(colors)], label=bidi.get_display(group))
        bottom += count[group]
    if logscale:
        ax.set_yscale('log')
    ax.legend()
    utils.draw()

def words_incidence_matrix(df, voc, col='text', per='article',
                           normalize=False, min_abs=0, min_perc=0, binary=False):
    # get list of texts
    data = list()
    sep = SEPARATOR[per]
    for txt in df[col]:
        data.extend([s.strip().strip(word_chars_filter) for s in
                     list(filter(None, re.split(sep, txt)))])
    # fill incidence matrix
    c = np.zeros(len(voc))
    D = np.zeros((len(voc), len(data)))
    for j,txt in enumerate(data):
        for w in get_all_words(
                txt, filter_fun=lambda w: any('א'<=c<='ת' for c in w)):
            if w in voc:
                c[voc.index(w)] += 1
                D[voc.index(w), j] += 1
    # normalize
    D[D<min_abs] = 0
    if normalize:
        D = D * np.nan_to_num(1/c)[:, np.newaxis]
        D[D<min_perc] = 0
    if binary: D[D>0] = 1
    return D

def words_local_incidence_matrix(df, voc, col='text', window=3,
                                 normalize=True, min_abs=0, min_perc=0.1, binary=False):
    full_voc = list(np.unique(get_all_words(df[col])))#, stopwords=tuple())))
    # get list of sentences
    data = get_all_sentences(df[col])
    # fill incidence matrix
    c = np.zeros(len(voc))
    D = np.zeros((len(voc), len(full_voc)))
    for txt in data:
        sent = get_all_words(
            txt, filter_fun=lambda w: any('א'<=c<='ת' for c in w))
        for k,w in enumerate(sent):
            if w in voc:
                i = voc.index(w)
                c[i] += 1
                neihb = sent[k-window:k] + sent[k+1:k+window+1]
                for w2 in neihb:
                    D[i, full_voc.index(w2)] += 1
    # normalize
    D[D<min_abs] = 0
    if normalize:
        D = D * np.nan_to_num(1/c)[:, np.newaxis]
        D[D<min_perc] = 0
    if binary: D[D>0] = 1
    return D

def words_2gram_adj_matrix(df, voc, col='text', window=2,
                           normalize=True, min_abs=0, min_perc=0.0, binary=False):
    full_voc = list(np.unique(get_all_words(
        df[col], filter_fun=lambda w: any('א'<=c<='ת' for c in w)
    )))
    # get list of sentences
    data = get_all_sentences(df[col])
    # fill incidence matrices
    c = np.zeros(len(voc))
    offsets = list(range(-window,0)) + list(range(1,window+1))
    D = {off: np.zeros((len(voc), len(full_voc))) for off in offsets}
    for txt in data:
        sent = get_all_words(
            txt, filter_fun=lambda w: any('א'<=c<='ת' for c in w))
        for k,w in enumerate(sent):
            if w in voc:
                i = voc.index(w)
                c[i] += 1
                for off in offsets:
                    if 0 <= k+off < len(sent):
                        D[off][i, full_voc.index(sent[k+off])] += 1
    # normalize
    for off in offsets:
        D[off][D[off]<min_abs] = 0
        if normalize:
            D[off] = D[off] * np.nan_to_num(1/c)[:, np.newaxis]
            D[off][D[off]<min_perc] = 0
        if binary: D[off][D[off]>0] = 1
    # adj matrix
    A = np.zeros((len(voc),len(voc)))
    for off in offsets:
        d = np.sqrt(D[off])
        A += np.matmul(d, d.transpose())
    np.fill_diagonal(A, 0)
    return (A, D, full_voc)

def graph_of_words(voc, df=None, D=None, A=None, filter_singletons=False, **kwargs):
    if A is None:
        if D is None:
            D = words_incidence_matrix(df, voc, **kwargs)
        D2 = np.sqrt(D)
        A = np.matmul(D2, D2.transpose())
        np.fill_diagonal(A, 0)
    if filter_singletons:
        ids = np.any(A != 0, axis=1)
        A = A[ids,:][:,ids]
        voc = [voc[i] for i,x in enumerate(ids) if x]
    G = nx.from_numpy_matrix(A)
    nx.relabel_nodes(G, {i: w for i, w in enumerate(voc)}, False)
    return G

def words2sections(G, df, to_plot=False):
    sections = np.unique(df.section)
    colors = {sec: utils.DEF_COLORS[i%len(utils.DEF_COLORS)]
              for i,sec in enumerate(sections)}
    for w in G.node:
        G.node[w]['section'] = \
            sections[np.argmax(['\n'.join(df[df.section==sec].text).count(w)
                                for sec in sections])]
    if to_plot:
        G2 = nx.relabel_nodes(
            G, {w: bidi.get_display(w) for w in G.node}, True)
        nx.draw(G2, with_labels=True, edge_color='pink',
                node_color=[colors[G2.node[w]['section']] for w in G2.node])

def common_context(df, words, col='text', window=2):
    if isinstance(words[0],str):
        words = (words,)
    sents = get_all_sentences(df[col])
    for pair in words:
        print("Words:\t", pair)
        A, D, voc = words_2gram_adj_matrix(df, pair, col=col, window=window)
        context = []
        for o in D:
            ii = [i[0] for i in np.argwhere(D[o][0,:] * D[o][1,:])]
            context.extend([voc[i] for i in ii])
        print("context:\t", context)
        for i, s in enumerate(sents):
            if np.any([w in s for w in pair]) and \
                    np.any([w in s for w in context]):
                print(i, s)

def word2vec(df, col='text', size=100, window=3,
             min_count=1, workers=4, save_to=None, **kwargs):
    sents = get_all_sentences(df[col])
    sents = [get_all_words(s,stopwords=()) for s in sents]
    model = Word2Vec(sents, size=size, window=window,
                     min_count=min_count, workers=workers, **kwargs)
    if save_to:
        pickle.dump(model, open(save_to,'wb'))
    return model

def tsne_plot(model, search_word, n_neighbors=10, ax=None):
    # Credit:
    # https://medium.com/@khulasaandh/word-embeddings-fun-with-word2vec-and-game-of-thrones-ea4c24fcf1b8
    labels = [bidi.get_display(search_word)]
    tokens = [model.wv[search_word]]
    similar = [1]
    close_words = model.wv.similar_by_word(search_word, topn=n_neighbors)
    for word in close_words:
        tokens.append(model.wv[word[0]])
        labels.append(bidi.get_display(word[0]))
        similar.append(word[1])

    tsne_model = TSNE(n_components=2, init='pca')
    coordinates = tsne_model.fit_transform(tokens)
    df = pd.DataFrame({'x': [x for x in coordinates[:, 0]],
                       'y': [y for y in coordinates[:, 1]],
                       'words': labels,
                       'similarity': similar}
                      )

    if ax is None:
        _, ax = plt.subplots()
    plot = ax.scatter(df.x, df.y, c=df.similarity, cmap='Reds')
    for i in range(len(df)):
        ax.annotate("  {} ({:.2f})".format(df.words[i].title(),
                                           df.similarity[i]),
                    (df.x[i], df.y[i]))

    plt.colorbar(mappable=plot, ax=ax)
    ax.set_title('t-SNE visualization for {}'.format(
        bidi.get_display(search_word)))


if __name__ == "__main__":
    t0 = time()
    df = ba.load_data(r'..\Data\articles')
    print(f'Data loaded ({time()-t0:.0f} [s]).')

    # # Words frequencies
    # all_words = get_all_words(df.text,
    #                           filter_fun=lambda w: any('א'<=c<='ת' for c in w))
    # tt = utils.table(all_words, -1)
    # print(f'Words:\t{len(all_words):d}')
    # print(f'Unique words:\t{len(tt):d}')
    # print('Most repeating words:')
    # pprint(tt[:10])
    # plot_words_repetitions(tt)
    # print(f'Words stats done ({time()-t0:.0f} [s]).')
    #
    # # Politics
    # fig,ax = plt.subplots(1,2)
    # count_parties(ax[0], df)
    # count_parties(ax[1], df, binary_per_text=True)
    # fig,ax = plt.subplots(1,2)
    # count_parties(ax[0], df, by='section')
    # count_parties(ax[1], df, by='section', binary_per_text=True)
    # print(f'Politic stats done ({time()-t0:.0f} [s]).')

    print("\n\n____________________")
    voc = get_vocabulary(df, required_freq=12)
    print(f"Vocabulary loaded ({len(voc):d} words) ({time()-t0:.0f} [s]).")
    voc = list(np.random.choice(voc, 120, replace=False))
    print(f"Vocabulary shrunk ({len(voc):d} words) ({time()-t0:.0f} [s]).")

    # Words that share articles
    G1 = graph_of_words(voc, D=words_incidence_matrix(df, voc, min_abs=4, binary=True))
    print(f'Graph of shared articles generated ({time()-t0:.0f} [s]).')
    utils.info(G1, verbose=1)
    print(f'Graph of shared articles diagnosed ({time()-t0:.0f} [s]).')
    words2sections(G1, df, True)
    # A = nx.adj_matrix(G1).todense()
    # y = SpectralClustering(10).fit(A)
    # fig, ax = plt.subplots(1)
    # ax.scatter([A[:,0]], [A[:,1]], c=y.labels_)

    # Words that share adjacent words
    G2 = graph_of_words(voc, D=words_local_incidence_matrix(
        df, voc, window=2, min_abs=3, min_perc=0.05, binary=True),
                        filter_singletons=True)
    # pickle.dump(G2, open('tmp_context_graph.pkl','wb'))
    print(f'Graph of shared context generated ({time()-t0:.0f} [s]).')
    utils.info(G2, verbose=1)
    print(f'Graph of shared context diagnosed ({time()-t0:.0f} [s]).')
    words2sections(G2, df, True)
    # A = nx.adj_matrix(G2).todense()
    # y = SpectralClustering(10).fit(A)
    # fig, ax = plt.subplots(1)
    # ax.scatter([A[:,0]], [A[:,1]], c=y.labels_)
    for c in nx.algorithms.clique.find_cliques(G2):
        if len(c) > 2: print(c)

    # Words that share adjacent words
    A, D, full_voc = words_2gram_adj_matrix(
        df, voc, window=2, normalize=True, min_abs=3, min_perc=0.05, binary=True)
    G3 = graph_of_words(voc, A=A, filter_singletons=True)
    print(f'Graph of 2-grams generated ({time()-t0:.0f} [s]).')
    utils.info(G3, verbose=1)
    print(f'Graph of 2-grams diagnosed ({time()-t0:.0f} [s]).')
    words2sections(G3, df, True)
    # A = nx.adj_matrix(G3).todense()
    # y = SpectralClustering(10).fit(A)
    # fig, ax = plt.subplots(1)
    # ax.scatter([A[:,0]], [A[:,1]], c=y.labels_)
    for c in nx.algorithms.clique.find_cliques(G3):
        if len(c) > 2: print(c)
    p90 = utils.dist([e['weight'] for es in G3.edge.values() for e in es.values()])[-1]
    es = np.unique([{w1,w2} for w1 in G3.edge for w2 in G3.edge[w1]
                    if G3.edge[w1][w2]['weight']>=p90])
    print(es)
    # common_context(df, [list(e) for e in es])

    # # Word2vec model
    # model = pickle.load(
    #     open(r'..\Output\Context based embedding\word2vec.pkl','rb'))
    # _, axs = plt.subplots(2, 2)
    # for i, w in enumerate(('נתניהו', 'ירקות', 'אפשר', 'מכבי')):
    #     tsne_plot(model, w, ax=axs[int(i/2), i%2])

    plt.show()


# TODO large-scale network visualization
# TODO clustering (connectivity / cliques / small-conductance / others)
# TODO plannar embedding (represent using eigenvectors of A) (P. 10)?
