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
Status:
- HTMLs are very structured with convenient and distinctive labels, which
makes life easier.
- However, most pages contents are blocked, and most HTMLs are different
between browser and urllib for reasons I don't know
(maybe different default flags in the GET requests).
'''

SECTIONS = (['https://www.haaretz.co.il/news',
             'https://www.haaretz.co.il/news/world',
             'https://www.haaretz.co.il/opinions',
             'https://www.haaretz.co.il/sport',
             'https://www.haaretz.co.il/food',
             'https://www.haaretz.co.il/news/elections',
             'https://www.haaretz.co.il/science',
             'https://www.themarker.com/'],
            ['חדשות', 'בעולם', 'דעות', 'ספורט', 'אוכל', 'בחירות 2019', 'מדע', 'כלכלה'])

def get_all_articles(urls, sections_titles, verbose=1, demo=False):
    articles = dict()
    for url,section in zip(urls,sections_titles):
        articles[section] = \
            get_section_articles(url, section, verbose=verbose, demo=demo)
    return articles

def get_section_articles(url, title=None, verbose=1, demo=False):
    # get articles
    themarker = url==r"https://www.themarker.com/"
    base_url = url if themarker else r"https://www.haaretz.co.il"
    source = BeautifulSoup(urlopen(url), 'lxml')
    articles = st.get_all_links_with_strings(
        source, (url+'/',url[len(base_url):]+'/'), str_requirement_fun=any)
    if demo:
        articles = articles[:3]
    urls = [a.get('href') for a in articles]
    if themarker:
        urls = [u for u in urls if u and (base_url in u or not 'http' in u)
                and re.findall('1\.[5-8][0-9]', u)]
    urls = st.relative_to_absolute_url(urls, base_url)
    # remove duplications
    urls, ids, duplications = st.remove_duplications(urls)
    titles = [None for _ in ids]
    # summary
    st.print_scrapped_articles_summary(urls, verbose=verbose,
                                       header=title, duplications=duplications)
    return (urls, titles)

def get_articles_data(sections, save_to=None,
                      update_freq=10, verbose=2, demo=False):
    return st.get_articles_data_from_sections(
        sections, get_article_data, save_to=save_to,
        sheet_name='haaretz_demo' if demo else 'haaretz',
        update_freq=update_freq, verbose=verbose
    )

def get_article_data(url):
    if 'https://www.themarker.com/' in url:
        return get_tm_article_data(url)
    # get page
    try:
        html = urlopen(url)
    except:
        warn(f'Bad URL: {url:s}')
        return 'BAD_URL'
    soup = BeautifulSoup(html, 'lxml')
    # get data from page
    try:
        title = soup.find_all('title')[0].text
        text = '\n'.join([par.text.strip()
                          for par in soup.find_all('p',class_='t-body-text')
                          if not 'רשימת הקריאה מאפשרת לך' in par.text and
                          not 'לחיצה על כפתור "שמור"' in par.text and
                          not 'שים לב: על מנת להשתמש ברשימת הקריאה' in par.text])
    except:
        warn(f'Could not get title and body: {url:s}')
        return 'IRRELEVANT_PAGE'
    if len(text) < 30:
        return 'IRRELEVANT_PAGE'
    try:
        subtitle = [s.get('content').strip() for s in soup.find_all('meta')
                    if s.get('name')=='description'][0]
    except:
        subtitle = None
    try:
        author = [a for a in soup.find_all('a')
                  if a.get('data-statutil-writer') is not None][0].text.strip()
    except:
        author = None
    try:
        date = [a for a in soup.find_all('time')
                if a.get('itemprop')=='datePublished'][0].text
        date = re.findall('[0-3][0-9]\.[0-1][0-9]\.20[0-2][0-9]', date)[0]
        date = datetime.strptime(date, '%d.%m.%Y').date()
    except:
        date = None

    return {'url':url, 'title':title, 'subtitle':subtitle,
            'author':author, 'date':date, 'text':text}

def get_tm_article_data(url):
    # get page
    try:
        html = urlopen(url)
    except:
        warn(f'Bad URL: {url:s}')
        return 'BAD_URL'
    soup = BeautifulSoup(html, 'lxml')
    # get data from page
    try:
        title = soup.find_all('h1')[0].text
        text = '\n'.join([par.text.strip()
                          for par in soup.find_all('p',class_='t-body-text')
                          if not 'רשימת הקריאה מאפשרת לך' in par.text and
                          not 'לחיצה על כפתור "שמור"' in par.text and
                          not 'שים לב: על מנת להשתמש ברשימת הקריאה' in par.text])
    except:
        warn(f'Could not get title and body: {url:s}')
        return 'IRRELEVANT_PAGE'
    if len(text) < 30:
        return 'IRRELEVANT_PAGE'
    try:
        subtitle = [s.text.strip() for s in soup.find_all('p')
                    if s.get('itemprop')=='description'][0]
    except:
        subtitle = None
    author = None
    try:
        date = [a for a in soup.find_all('time')
                if a.get('itemprop')=='datePublished'][0].text
        date = re.findall('[0-3][0-9]\.[0-1][0-9]\.20[0-2][0-9]', date)[0]
        date = datetime.strptime(date, '%d.%m.%Y').date()
    except:
        date = None

    return {'url':url, 'title':title, 'subtitle':subtitle,
            'author':author, 'date':date, 'text':text}


if __name__=='__main__':
    demo = False
    urls,sections = SECTIONS
    print("\nGetting articles urls...")
    articles = get_all_articles(urls, sections, demo=demo)
    print("\nScrapping articles...")
    df = get_articles_data(articles, demo=demo,
                           save_to=r'd:\code\python\news\scrapper\articles')
    print('Done.')
