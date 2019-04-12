import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import networkx as nx
from scipy.linalg import eigh
import re
import time
from pprint import pprint
from warnings import warn
from datetime import datetime
import itertools
import bidi.algorithm as bidi
import matplotlib.dates as mdates
import matplotlib.ticker as ticker


##################   STATS UTILS   ##################

def dist(x, quantiles=(0,10,50,90,100), do_round=False):
    '''
    Get the distribution of a vector as quantiles + very basic stats.
    :param x: numeric numpy array.
    :param quantiles: quantiles in [0,100] to find.
    :param do_round: whether to round numbers or not.
    :return: a list [len(x), mean(x), quant1(x), quant2(x), ..., quantn(x)].
    '''
    if not len(x):
        return [None for _ in range(2+len(quantiles))]
    s = [len(x), np.mean(x)] + list(np.percentile(x,quantiles))
    return [int(z+np.sign(z)*0.5) for z in s] if do_round else s

def table(l, sort=0):
    '''
    :param l: iterable object.
    :param sort: 0 for no sort; +-1 for increasing/decreasing frequency respectively.
    :return: tuple of 2D-tuples (value, number of occurences in l).
    '''
    try:
        # if it can be sorted, it should (otherwise groupby may work unexpectedly).
        tab = tuple((g[0], len(list(g[1])))
                    for g in itertools.groupby(sorted(l)))
    except:
        # if it can't be sorted (e.g. datetime), try groupby anyway.
        tab = tuple((g[0], len(list(g[1])))
                    for g in itertools.groupby(l))
    if sort:
        tab = sorted(tab, key = lambda x: np.sign(sort)*x[1])
    return tab

def info(x, **kwargs):
    if isinstance(x, pd.DataFrame): info_data_frame(x, **kwargs)
    elif isinstance(x, nx.Graph): info_graph(x, **kwargs)
    else: print(f'Unsupported type: {type(x):s}')

def info_data_frame(df, verbose=1):
    assert(isinstance(df,pd.DataFrame)),\
        f'Bad type (not a DataFrame): {type(df):s}'
    print(f'\n\nData-Frame information:')
    # data frame info
    print(f'Dimensions: \t{df.shape[0]} X {df.shape[1]}')
    print('Columns:', end='\n\t')
    for i,col in enumerate(df.columns):
        print(col, end =
        '\n\t' if i+1==df.shape[1] or (i+1)%3==0 else '  |  ')
    print('')
    # columns stats
    if verbose >= 1:
        for col in df.columns:
            print(f'{col:s}:\n\tTypes: \t', end='')
            types = table([str(type(x))[8:-2] for x in df[col]], -1)
            for t in types:
                print(f'{t[0]:s} ({100*t[1]/len(df):.0f}%), ', end='')
            print('')
            print(f'\tUnique values: \t{len(np.unique(df[col])):d}')
            print('\tFrequent values: \t', end='')
            tab = table(df[col],-1)
            for i,t in enumerate(tab[:4]):
                try:
                    if len(t[0])>30:
                        tab[i] = (str(t[0][:30])+'...', tab[i][1])
                except:
                    pass
            print(*[f'{t[0]}'+f' ({t[1]:d}),' for t in tab[:4]])
            print('\tFrequent frequencies: \t', end='')
            tab = table([t[1] for t in tab], -1)
            print(*[f'{t[0]:d}'+f' ({t[1]:d}),' for t in tab[:4]])
            qs = (0, 10, 50, 90, 100)
            try:
                d = dist(df[col])
                print(f'\tAverage: \t{d[1]}')
                print('\tQuantiles: \t', end='')
                print(*[f'{q:.0f}%: {v}' for q, v in zip(qs, d[2:])],
                      sep=',   ')
            except:
                try:
                    d = dist([len(s) for s in df[col]])
                    print(f'\tAverage length: \t{d[1]}')
                    print('\tQuantiles: \t', end='')
                    print(*[f'{q:.0f}%: {v:.0f}'
                            for q, v in zip(qs, d[2:])],
                          sep=',   ')
                except:
                    pass
    # sample of contents
    if verbose >= 2:
        if (len(df)<20):
            print(df)
        else:
            print('\nHead:')
            print(df.head(5))
            print('\nMid:')
            print(df[int(len(df)/2)-2:int(len(df)/2)+2])
            print('\nTail:')
            print(df.tail(5))
        print('')
    # add optional plots if verbose >= 3?

def info_graph(G, verbose=1):
    print(f'\n\nGraph information:')
    n = G.number_of_nodes()
    if n==0:
        print('Empty graph.')
        return
    m = G.number_of_edges()
    self_loops = G.number_of_selfloops()
    print(f'Nodes: {n:d}\t\tEdges: {m:d}\t\t' +
          f'Density: {100*(m-self_loops)/sp.special.binom(n,2):.0f}%\t\t' +
          f'Self-loops: {100*self_loops/n:.0f}%')
    CC = nx.algorithms.components.number_connected_components(G)
    C = max(nx.connected_component_subgraphs(G), key=len)
    print(f'Connectivity components: {CC:d}\t\t' +
          f'Largest component: {len(C):d} nodes')
    print('Clustering coefficient:\t{0:.0f}%'.format(
        100*nx.algorithms.cluster.average_clustering(G)))
    print('Distribution of degrees:')
    print(*[0,10,50,90,100], sep='%\t')
    print(dist(list(G.degree().values()), do_round=True)[2:])
    if verbose >= 2:
        print('Largest component diameter:\t{0:d}'.format(
            nx.algorithms.distance_measures.diameter(C) ))
        # Spectrum
        L = nx.laplacian_matrix(G)
        x, v = eigh(L.todense())
        print('Spectral-estimated number of clusters:\t{0:d}'.format(
            np.argmax(np.diff(x)) + 1 ))
        if n <= 100:
            fig, ax = plt.subplots(1, 1)
            barplot(ax, list(range(n)), x, ylab='Eigenvalue', colors='b')
            draw()
        else:
            print('Distribution of Laplacian spectrum:')
            print(*[0,10,50,90,100], sep='%\t')
            print(dist(x)[2:])
        if verbose >= 3:
            if n <= 1000:
                fig, ax = plt.subplots(1, 1)
                plt.axes(ax)
                Gtmp = nx.relabel_nodes(
                    G, {w: bidi.get_display(w) for w in G.node}, True)
                nx.draw(Gtmp, with_labels=True, edge_color='green')
                if verbose >= 4:
                    # spectral embedding of both Graph & largest component
                    fig, ax = plt.subplots()
                    nx.draw(Gtmp, pos={w: (v[i, 1], v[i, 2])
                                    for i,w in enumerate(Gtmp.node.keys())},
                            with_labels=True, edge_color='green')
                    Ctmp = max(nx.connected_component_subgraphs(Gtmp), key=len)
                    fig, ax = plt.subplots()
                    nx.draw(Ctmp, pos={w: (v[i, 1], v[i, 2])
                                    for i,w in enumerate(Ctmp.node.keys())},
                            with_labels=True, edge_color='green')

def count(txt, sep):
    '''
    Count tokens in text.
    :param txt: string or iterable of strings.
    :param sep: separator for text.
    :return: count of tokens or list of counts.
    '''
    if isinstance(txt,str):
        return len(list(filter(None,re.split(sep,txt))))
    else:
        return [len(list(filter(None,re.split(sep,s)))) for s in txt]

##################   PLOT UTILS   ##################

def draw():
    plt.get_current_fig_manager().window.showMaximized()
    plt.draw()
    plt.pause(1e-10)
    plt.tight_layout()

def clean_figure(ax):
    '''
    Clean an empty figure (remove x,y axes and ticks).
    '''
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_xticklabels(())
    ax.set_yticklabels(())

DEF_COLORS = ('blue','red','green','purple','orange','grey','pink')

def barplot(ax, x, y, bottom=None, plot_bottom=True,
            title=None, xlab=None, ylab=None, xlim=None, ylim=None, label=None,
            vertical_xlabs=False, colors=DEF_COLORS, bcolors=DEF_COLORS):
    '''
    pyplot's barplot wrapper.
    '''
    xticks = None
    if any((isinstance(xx,str) for xx in x)):
        xnames = x
        x = tuple(range(len(x)))
        xticks = x
        xlim = (-0.5,len(xnames)-0.5)
    if bottom is not None and plot_bottom:
        ax.bar(x, bottom,
               color=bcolors[:len(x)] if isinstance(bcolors,tuple) else bcolors)
    ax.bar(x, y, bottom=bottom, label=label,
           color=colors[:len(x)] if isinstance(colors,tuple) else colors)
    if title: ax.set_title(title, fontsize=14)
    if xlab: ax.set_xlabel(xlab, fontsize=12)
    if ylab: ax.set_ylabel(ylab, fontsize=12)
    if xlim: ax.set_xlim(xlim[0],xlim[1])
    if ylim: ax.set_ylim(ylim[0],ylim[1])
    if xticks:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xnames, fontsize=10)
        if vertical_xlabs:
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
