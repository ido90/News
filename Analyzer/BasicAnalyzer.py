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
import Scrapper.ScrapperTools as st

def load_data(path,
              sheets=('ynet', 'mako', 'haaretz'),
              filter_str=('source','title','text'),
              verbose=1):
    df = st.load_data_frame(path, sheets=sheets, verbose=verbose)
    for h in filter_str:
        df = df[[(isinstance(t, str) and len(t)>0) for t in df[h].values]]
    df['blocked'] = [src=='haaretz' and txt.endswith('...')
                     for src,txt in zip(df['source'], df['text'])]
    return df

############## GENERAL TOOLS ##############

DEF_COLORS = ('blue','red','green','purple','orange','grey','pink')

def dist(x, quantiles=(0,10,50,90,100), do_round=False):
    if not len(x):
        return [None for _ in range(2+len(quantiles))]
    s = [len(x), np.mean(x)] + list(np.percentile(x,quantiles))
    return [int(z+np.sign(z)*0.5) for z in s] if do_round else s

def barplot(ax, x, y, bottom=None, plot_bottom=True,
            title=None, xlab=None, ylab=None, xlim=None, ylim=None, label=None,
            vertical_xlabs=False, colors=DEF_COLORS, bcolors=DEF_COLORS):
    xticks = None
    if any((isinstance(xx,str) for xx in x)):
        xnames = x
        x = tuple(range(len(x)))
        xticks = x
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

def bar_per_source(ax, df, ylab, fun, title=None, ylim=None,
                   colors='black', bcolors=DEF_COLORS):
    sources = np.unique(df.source)
    barplot(ax, sources,
            [fun(df[np.logical_and(df.source==src,df.blocked)]) for src in sources],
            bottom=
            [fun(df[np.logical_and(df.source==src,np.logical_not(df.blocked))])
             for src in sources],
            ylab=ylab, title=title, colors=colors, bcolors=bcolors)

def clean_figure(ax):
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_xticklabels(())
    ax.set_yticklabels(())

############## ANALYSIS ##############

def data_description(df):
    sources = np.unique(df['source'])
    n = len(sources)
    f, axs = plt.subplots(2, n)
    # counters per source
    bar_per_source(axs[0,0], df, 'Articles\n(black = partially blocked contents)',
                   lambda d: d.shape[0], '\nArticles per Source')
    bar_per_source(axs[0,1], df,
                   'Words [x1000]\n(black = partially blocked contents)',
                   lambda d: sum(len(l.split()) for t in d['text'].values
                                 for l in t.split('\n')) / 1e3,
                   'BASIC DATA DESCRIPTION\nWords per Source')
    # remove blocked haaretz texts before analysis
    df = df[np.logical_not(df['blocked'])]
    # sections per source
    articles_per_section =\
        [df[np.logical_and(df.source==src,df.section==sec)].shape[0]
         for src in sources
         for sec in np.unique(df[df.source==src].section)]
    axs[0,2].pie([df[df.source==src].shape[0] for src in sources],
                 labels=sources, colors=DEF_COLORS[:3], startangle=90,
                 frame=True, counterclock=False)
    patches,_ = axs[0,2].pie(articles_per_section,
                  radius=0.75, startangle=90, counterclock=False)
    centre_circle =\
        plt.Circle((0, 0), 0.5, color='black', fc='white', linewidth=0)
    axs[0,2].add_artist(centre_circle)
    axs[0,2].set_title('\nSources and Sections', fontsize=14)
    axs[0,2].legend(
        patches, [bidi.get_display(sec) for src in sources
                  for sec in np.unique(df[df.source==src].section)],
        ncol=5, loc='upper right', bbox_to_anchor=(1, 0.11), fontsize=8 )
    # dates & authors
    date_hist(axs[1,0], df)
    author_concentration(axs[1,1], df)
    top_authors(axs[1,2], df)
    # draw
    plt.get_current_fig_manager().window.showMaximized()
    plt.draw()
    plt.pause(1e-17)

def date_hist(ax, df, old_thresh=np.datetime64(datetime(2019,3,1))):
    dts = [str(dt) if str(dt)=='NaT'
           else str(dt)[:4] if dt<old_thresh else str(dt)[:10]
           for dt in df.date]
    dts_vals = sorted(list(set(dts)))
    sources = np.unique(df.source)
    date_count = {src: [np.sum(sc==src and dt==dt_val for sc,dt in zip(df.source,dts))
                        for dt_val in dts_vals]
                  for src in sources}
    bottom = np.array([0 for _ in dts_vals])
    for i,src in enumerate(sources):
        barplot(ax, dts_vals, date_count[src], bottom=bottom, title='Dates',
                ylab='Articles', vertical_xlabs=True, label=src,
                colors=('b','r','g')[i], plot_bottom=False)
        bottom += date_count[src]
    ax.legend(loc='upper left')

def author_concentration(ax1, df):
    n = 0
    for k,src in enumerate(np.unique(df.source)):
        # calculate
        d = df[df.source==src]
        authors = np.array(sorted(list(set([str(a) for a in d.author]))))
        arts_per_aut = np.array([np.sum(d.author==a) for a in authors])
        ids = sorted(range(len(arts_per_aut)),
                     key=lambda i: arts_per_aut[i], reverse=True)
        authors = authors[ids]
        arts_per_aut = arts_per_aut[ids]
        arts_per_aut = np.cumsum(arts_per_aut)
        n = max(n,len(authors))
        # plot
        ax1.plot(list(range(len(arts_per_aut))), 100*arts_per_aut/d.shape[0],
                ('b-','r-','g-')[k], label=src)
        ax1.set_title('Authors', fontsize=14)
        ax1.set_xlabel('K', fontsize=12)
        ax1.set_ylabel(
            'Number of articles by most active K authors\n'+
            '(not reaching 100% due to unknown authors)', fontsize=12)
    ax1.set_xlim((0,n))
    ax1.set_ylim((0,100))
    ax1.legend()

def top_authors(ax, df, n=5):
    sources = np.unique(df.source)
    top_authors = {}
    top_authors_arts = {}
    for k,src in enumerate(sources):
        # calculate
        d = df[df.source==src]
        authors = np.array(sorted(list(set([str(a) for a in d.author]))))
        arts_per_aut = np.array([np.sum(d.author==a) for a in authors])
        ids = sorted(range(len(arts_per_aut)),
                     key=lambda i: arts_per_aut[i], reverse=True)
        top_authors[src] = authors[ids[:n]]
        top_authors_arts[src] = arts_per_aut[ids[:n]]
    # plot
    width = 1/(n+1)
    for i in range(n):
        rects = ax.bar(
            np.arange(len(sources))+i*width,
            [top_authors_arts[src][i] for src in sources],
            width
        )
        for rect,src in zip(rects,sources):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., height+0.5,
                    f'{bidi.get_display(top_authors[src][i]):s}',
                    ha='center', va='bottom', rotation=90)
    ax.set_ylabel('Articles', fontsize=12)
    ax.set_xlabel('Top Authors', fontsize=12)
    ax.set_xticks(np.arange(len(sources)) + n*width/2)
    ax.set_xticklabels(sources)

def validity_tests(df):
    sources = np.unique(df['source'])
    blocked_contents = (1-check_haaretz_blocked_text(df[df['source'] == 'haaretz'])\
                       / np.sum(df['source']=='haaretz')) * 100
    df = df[np.logical_not(df['blocked'])]
    n = {src: np.sum(df['source'] == src) for src in sources}
    # get anomalies
    bad_types = {src: verify_types(df[df['source']==src],
                                   {'date':datetime,'blocked':np.bool_})
                 for src in sources}
    bad_lengths = {src: check_lengths(df[df['source']==src]) for src in sources}
    # plot anomalies
    f, axs = plt.subplots(2, len(sources))
    for i, src in enumerate(sources):
        tit = ('DATA VALIDITY TESTS\n' if i==int(len(sources)/2) else '\n') +\
              f'[{src:s}] Invalid field types' +\
              (f'\n(out of {blocked_contents:.0f}% unblocked articles)'
               if src=='haaretz' else '\n')
        barplot(axs[0, i], bad_types[src].keys(),
                100 * np.array(tuple(bad_types[src].values())) / n[src],
                vertical_xlabs=True, title=tit,
                ylab='Having invalid type [%]', ylim=(0, 100))
    sp = inspect.getfullargspec(check_lengths)
    limits = list(itertools.chain.from_iterable(sp[3][0].values()))
    for i, src in enumerate(sources):
        barplot(axs[1, i],
                [a+f'\n({b:.0f} chars)' for a,b in
                 zip(bad_lengths[src].keys(),limits)],
                100 * np.array(tuple(bad_lengths[src].values())) / n[src],
                vertical_xlabs=True,
                title=f'[{src:s}] Suspicious string-field lengths',
                ylab='Having invalid length [%]', ylim=(0, 100))
    # draw
    plt.get_current_fig_manager().window.showMaximized()
    plt.draw()
    plt.pause(1e-17)

def verify_types(df, types=(), default_type=str):
    bad_type = {}
    for col in df.columns:
        tp = types[col] if col in types else default_type
        bad_type[col] = np.sum([not isinstance(x, tp) for x in df[col]])
    return bad_type

def check_lengths(df, lengths={'section': (2, 20), 'title': (10, 6 * 30),
                               'subtitle': (10, 6 * 70), 'date': (6, 12),
                               'author': (2, 30), 'text': (6 * 60, np.inf)}):
    exceptional_length = {}
    for l in lengths:
        exceptional_length['short_'+l] =\
            np.sum([isinstance(s,str) and len(s)<lengths[l][0] for s in df[l]])
        exceptional_length['long_'+l]  =\
            np.sum([isinstance(s,str) and len(s)>lengths[l][1] for s in df[l]])
    return exceptional_length

def check_haaretz_blocked_text(df):
    assert (all(src == 'haaretz' for src in df['source']))
    return np.sum([s.endswith('...') for s in df['text']])


# TODO counters:
# boxplot(source, n_words per title)
# boxplot(source, n_words per subtitle)
# boxplot(source, n_words per text)
# boxplot(source, n_chars per title)
# boxplot(source, n_chars per subtitle)
# boxplot(source, n_chars per text)
# boxplot(source, n_chars per text_word)
# boxplot(source, 100*n_unique_chars/n_chars per text)

# per section basic analysis
# TODO same as per source? maybe generalize functions to receive sources/sections as input.

############## MAIN ##############

if __name__ == "__main__":
    df = load_data(r'D:\Code\Python\News\Scrapper\articles')
    data_description(df.copy())
    validity_tests(df.copy())
    plt.tight_layout()
    plt.show()
