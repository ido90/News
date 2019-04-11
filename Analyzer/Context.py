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
import general_utils.utils as utils
import Scrapper.ScrapperTools as st
import Analyzer.BasicAnalyzer as ba
import Analyzer.Semantics as sm


#################   CONTEXT-BASED GRAPH REPRESENTATION   #################

def words_incidence_matrix(df, voc, col='text', per='article',
                           normalize=False, min_abs=0, min_perc=0, binary=False):
    # get list of texts
    data = list()
    sep = sm.SEPARATOR[per]
    for txt in df[col]:
        data.extend([s.strip().strip(sm.word_chars_filter) for s in
                     list(filter(None, re.split(sep, txt)))])
    # fill incidence matrix
    c = np.zeros(len(voc))
    D = np.zeros((len(voc), len(data)))
    for j,txt in enumerate(data):
        for w in sm.get_all_words(
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
    full_voc = list(np.unique(sm.get_all_words(df[col])))
    # get list of sentences
    data = sm.get_all_sentences(df[col])
    # fill incidence matrix
    c = np.zeros(len(voc))
    D = np.zeros((len(voc), len(full_voc)))
    for txt in data:
        sent = sm.get_all_words(
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
    full_voc = list(np.unique(sm.get_all_words(
        df[col], filter_fun=lambda w: any('א'<=c<='ת' for c in w)
    )))
    # get list of sentences
    data = sm.get_all_sentences(df[col])
    # fill incidence matrices
    c = np.zeros(len(voc))
    offsets = list(range(-window,0)) + list(range(1,window+1))
    D = {off: np.zeros((len(voc), len(full_voc))) for off in offsets}
    for txt in data:
        sent = sm.get_all_words(
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

def graph_of_words(voc, df=None, D=None, A=None, A_thresh=0,
                   filter_singletons=False, **kwargs):
    if A is None:
        if D is None:
            D = words_incidence_matrix(df, voc, **kwargs)
        D2 = np.sqrt(D)
        A = np.matmul(D2, D2.transpose())
        np.fill_diagonal(A, 0)
        A[A<A_thresh] = 0
    if filter_singletons:
        ids = np.any(A != 0, axis=1)
        A = A[ids,:][:,ids]
        voc = [voc[i] for i,x in enumerate(ids) if x]
    G = nx.from_numpy_matrix(A)
    nx.relabel_nodes(G, {i: w for i, w in enumerate(voc)}, False)
    return G

def words2sections(G, df, to_plot=False, title=''):
    sections = np.unique(df.section)
    colors = ('cyan', 'red', 'green', 'lime', 'orange', 'gold',
              'grey', 'magenta', 'plum', 'peru')
    # assign to each word the section in which it appears the most
    for w in G.node:
        G.node[w]['section'] = \
            sections[np.argmax(['\n'.join(df[df.section==sec].text).count(w)
                                for sec in sections])]
    sections = np.unique([G.node[w]['section'] for w in G.node])
    if to_plot:
        colors = {sec: colors[i % len(colors)]
                  for i, sec in enumerate(sections)}
        G2 = nx.relabel_nodes(
            G, {w: bidi.get_display(w) for w in G.node}, True)
        # nx.draw(G2, with_labels=True, edge_color='pink',
        #         node_color=[colors[G2.node[w]['section']] for w in G2.node])
        pos = nx.spring_layout(G2)
        for sec in sections:
            nx.draw_networkx_nodes(G2, pos=pos, node_color=colors[sec],
                                   with_labels=True, label=bidi.get_display(sec),
                                   nodelist=[w for w in G2
                                             if G2.node[w]['section']==sec])
        nx.draw_networkx_edges(G2, pos=pos, edge_color='pink')
        nx.draw_networkx_labels(G2, pos=pos)
        plt.title(title)
        plt.legend()
        utils.draw()

def common_context(df, words, col='text', window=2):
    if isinstance(words[0],str):
        words = (words,)
    sents = sm.get_all_sentences(df[col])
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


#################   WORD2VEC   #################

def word2vec(df, col='text', size=100, window=3,
             min_count=1, workers=4, save_to=None, **kwargs):
    sents = sm.get_all_sentences(df[col])
    sents = [sm.get_all_words(s,stopwords=()) for s in sents]
    model = Word2Vec(sents, size=size, window=window,
                     min_count=min_count, workers=workers, **kwargs)
    if save_to:
        pickle.dump(model, open(save_to,'wb'))
    return model

def tsne_plot(model, search_word, n_neighbors=10, ax=None):
    # Credit for function:
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
    utils.draw()


#################   MAIN   #################

if __name__ == "__main__":
    ## configuration
    build_graphs = False
    save_graphs = False
    voc_samples = 0
    detailed_cliques = False
    build_word2vec = False

    # load data
    t0 = time()
    df = ba.load_data(r'..\Data\articles')
    print(f'Data loaded ({time()-t0:.0f} [s]).')

    # get vocabulary
    print("\n\n____________________")
    voc = sm.get_vocabulary(df, required_freq=70,
                            filter_fun=lambda w: any('א' <= c <= 'ת' for c in w))
    print(f"Vocabulary loaded ({len(voc):d} words) ({time()-t0:.0f} [s]).")
    if voc_samples > 0:
        voc = list(np.random.choice(voc, voc_samples, replace=False))
        print(f"Vocabulary shrunk ({len(voc):d} words) ({time()-t0:.0f} [s]).")

    # Words that share articles
    # G1 = graph_of_words(voc, D=words_incidence_matrix(df, voc, min_abs=2, binary=True))
    # print(f'Graph of shared articles generated ({time()-t0:.0f} [s]).')
    # utils.info(G1, verbose=1)
    # print(f'Graph of shared articles diagnosed ({time()-t0:.0f} [s]).')
    # fig,ax = plt.subplots()
    # words2sections(G1, df, True)
    # A = nx.adj_matrix(G1).todense()
    # y = SpectralClustering(10).fit(A)
    # fig, ax = plt.subplots(1)
    # ax.scatter([A[:,0]], [A[:,1]], c=y.labels_)

    # Graph of shared skip-grams neighbors
    if build_graphs:
        A, D, full_voc = words_2gram_adj_matrix(
            df, voc, window=3, normalize=True, min_abs=3, min_perc=0.0, binary=True)
        G = graph_of_words(voc, A=A, filter_singletons=True, A_thresh=2)
        if save_graphs:
            pickle.dump(G, open(r'..\Output\Context based embedding\2gram_based_graph.pkl', 'wb'))
    else:
        G = pickle.load(open(r'..\Output\Context based embedding\2gram_based_graph.pkl', 'rb'))
    print(f'Graph of 2-grams generated ({time()-t0:.0f} [s]).')
    utils.info(G, verbose=1)
    print(f'Graph of 2-grams diagnosed ({time()-t0:.0f} [s]).')
    fig,ax = plt.subplots()
    words2sections(G, df, to_plot=True, title='2-gram context based similarities')
    # TODO
    # A = nx.adj_matrix(G).todense()
    # y = SpectralClustering(10).fit(A)
    # fig, ax = plt.subplots(1)
    # ax.scatter([A[:,0]], [A[:,1]], c=y.labels_)
    # Cliques
    if detailed_cliques:
        for c in nx.algorithms.clique.find_cliques(G):
            if len(c) > 2: print(c)
    p90 = utils.dist([e['weight'] for es in G.edge.values() for e in es.values()])[-1]
    large_weight_edges = np.unique([{w1,w2} for w1 in G.edge for w2 in G.edge[w1]
                    if G.edge[w1][w2]['weight']>=p90])
    print('Edges with weight > quantile-90%:')
    print(large_weight_edges)
    if detailed_cliques:
        common_context(df, [list(e) for e in large_weight_edges])

    # Word2vec model
    if build_word2vec:
        model = word2vec(df, save_to=r'..\Output\Context based embedding\word2vec.pkl')
    else:
        model = pickle.load(
            open(r'..\Output\Context based embedding\word2vec.pkl','rb'))
    _, axs = plt.subplots(2, 2)
    for i, w in enumerate(('נתניהו', 'ירקות', 'אפשר', 'מכבי')):
        tsne_plot(model, w, ax=axs[int(i/2), i%2])

    plt.show()


# TODO clustering (connectivity / cliques / small-conductance / others)
# TODO plannar embedding (represent using eigenvectors of A) (P. 10)?
