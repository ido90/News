import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
from pprint import pprint
from warnings import warn
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from urllib.request import urlopen
from bs4 import BeautifulSoup
import Scrapper.ScrapperTools as st

'''
Status: scrapping seems to be quite successful,
with ~200 articles ordered by sections.
'''

def get_homepage(url="https://www.mako.co.il"):
    return st.get_homepage(url)

def get_all_sections(home, verbose=1):
    # get sections
    sections = st.get_all_links_with_strings(home, ('?partner=NavBar',), True)
    urls = ['https://www.mako.co.il/news?partner=NavBar'] +\
           [s.get('href') for s in sections]
    titles = ['חדשות'] + [s.text.strip() for s in sections]
    # complete relative urls
    urls = st.relative_to_absolute_url(urls, r"https://www.mako.co.il")
    # filter undesired sections
    desired_sections = ('חדשות','כסף','ספורט','תרבות',
                        'בריאות','משפט','אוכל','לימודים וקריירה')
    ids = [i for i,tit in enumerate(titles) if tit.strip() in desired_sections]
    titles = [titles[i] for i in ids]
    urls = [urls[i] for i in ids]
    # remove duplications
    urls, ids, duplications = st.remove_duplications(urls)
    titles = [titles[i] for i in ids]
    # return
    if verbose >= 1:
        print(f"{len(titles):d} Sections:")
        pprint(titles)
    return (urls, titles)

def get_all_articles(urls, sections_titles, verbose=1, demo=False):
    articles = dict()
    for url,section in zip(urls,sections_titles):
        articles[section] = \
            get_section_articles(BeautifulSoup(urlopen(url),'lxml'),
                                 section, verbose=verbose, demo=demo)
    return articles

def get_section_articles(source, title=None, verbose=1, demo=False):
    # get articles
    articles = st.get_all_links_with_strings(source, ('/Article-',), True)
    if demo:
        articles = articles[:3]
    urls = [a.get('href') for a in articles]
    urls = st.relative_to_absolute_url(urls, r"https://www.mako.co.il")
    titles = [a.text.strip() for a in articles]
    titles = [tit[:tit.find('\r\n')] if tit.find('\r\n')!=-1 else tit
              for tit in titles]
    # remove duplications
    urls, ids, duplications = st.remove_duplications(urls)
    titles = [titles[i] for i in ids]
    # summary
    st.print_scrapped_articles_summary(urls, titles, verbose,
                                       header=title, duplications=duplications)
    return (urls, titles)

def get_articles_data(sections, save_to=None,
                      update_freq=10, verbose=2, demo=False):
    return st.get_articles_data_from_sections(
        sections, get_article_data, save_to=save_to,
        sheet_name='mako_demo' if demo else 'mako',
        update_freq=update_freq, verbose=verbose
    )

def get_article_data(url):
    # get page
    try:
        soup = st.url2html(url,3,False)
    except:
        warn(f'Bad URL: {url:s}')
        return 'BAD_URL'
    # get data from page
    try:
        title = soup.find_all('h1', class_=None)[0].text
        heb_letters = tuple(range(ord('א'), ord('ת') + 1))
        text = '\n'.join(
            [par.text.strip()
             for par in soup.find('div',class_='article').find_all('p', class_=None)
             if len(par.text.strip()) > 30
             and np.mean([ord(c) in heb_letters for c in par.text.strip()]) > 0.5]
        )
        if len(text) < 30: raise ValueError()
    except:
        warn(f"Could not process URL: {url:s}")
        return 'IRRELEVANT_PAGE'
    try:
        subtitle = soup.find_all('h2',class_=None)[0].text
    except:
        subtitle = None
    try:
        author = soup.find_all('span', class_='katav')[0].text
    except:
        author = None
    try:
        date = soup.find_all('span', class_='displayDate')[0].text
        date = re.findall('[0-3][0-9]/[0-1][0-9]/[0-2][0-9]', date)[0]
        date = datetime.strptime(date, '%d/%m/%y').date()
    except:
        date = None

    return {'url':url, 'title':title, 'subtitle':subtitle,
            'author':author, 'date':date, 'text':text}


if __name__=='__main__':
    demo = False
    print("Getting homepage...")
    home = get_homepage()
    print("\nGetting all sections")
    urls,sections = get_all_sections(home)
    print("\nGetting articles urls...")
    articles = get_all_articles(urls, sections, demo=demo)
    print("\nScrapping articles...")
    df = get_articles_data(articles, demo=demo,
                           save_to=r'd:\code\python\news\scrapper\articles')
    print('Done.')
