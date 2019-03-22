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
Status: unfortunately, most requests returned empty HTMLs. Not sure why
(maybe non-deterministic prevention of many requests from single IP?).
'''

def get_homepage(url="https://www.israelhayom.co.il"):
    return st.get_homepage(url)

def get_all_sections(home, verbose=1):
    # get sections
    # example: <a href="https://www.israelhayom.co.il/news" title="">חדשות</a>
    sections = st.get_all_links_with_strings(
        home, ('href="https://www.israelhayom.co.il/','" title="">'), True
    )
    urls = [s.get('href') for s in sections]
    titles = [s.text for s in sections]
    # filter undesired sections
    desired_sections = ('בחירות 2019','חדשות','דעות','כלכלה',
                        'ספורט','תרבות','אוכל','טכנולוגיה',
                        'בריאות','צרכנות','יחסים','קריירה')
    ids = [i for i,tit in enumerate(titles)
           if tit.strip() in desired_sections and
           not "search?section" in urls[i]]
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
            get_section_articles(st.url2html(url,5,False,2),
                                 section, verbose=verbose, demo=demo)
    return articles

def get_section_articles(source, title=None, verbose=1, demo=False):
    # get articles
    articles = st.get_all_links_with_strings(
        source, ('/article/','/opinion/'), True, str_requirement_fun=any)
    if demo:
        articles = articles[:3]
    urls = [a.get('href') for a in articles]
    urls = st.relative_to_absolute_url(urls, r"https://www.israelhayom.co.il")
    titles = [None for _ in articles]
    # remove duplications
    urls, ids, duplications = st.remove_duplications(urls)
    titles = [titles[i] for i in ids]
    # summary
    st.print_scrapped_articles_summary(urls, verbose=verbose,
                                       header=title, duplications=duplications)
    return (urls, titles)

def get_articles_data(sections, save_to=None,
                      update_freq=10, verbose=2, demo=False):
    return st.get_articles_data_from_sections(
        sections, get_article_data, save_to=save_to,
        sheet_name='IsraelHayom_demo' if demo else 'IsraelHayom',
        update_freq=update_freq, verbose=verbose
    )

def get_article_data(url):
    # get page
    try:
        soup = st.url2html(url, error_on_failure=False)
    except:
        warn(f'Bad URL: {url:s}')
        return 'BAD_URL'
    # get data from page
    heb_letters = tuple(range(ord('א'),ord('ת')+1))
    try:
        title = soup.find_all(class_='pane-title')[0].text
        pars = soup.find_all('div', class_='field-items')[2].\
            find('div', class_='field-item').find_all('div', class_=None)
        text = '\n'.join(
            [par.text.strip() for par in pars
             if len(par.text.strip())>30
             and np.mean([ord(c) in heb_letters for c in par.text.strip()])>0.5
             and 'העדכונים הכי חמים ישירות לנייד' not in par.text]
        )
    except:
        warn(f"Could not process URL: {url:s}")
        #return 'IRRELEVANT_PAGE'
        raise
    subtitle = soup.find_all(class_='field-item')
    subtitle = subtitle[0].text if subtitle else None

    author_date = soup.find_all(class_='art_header_footer_author')
    try:
        author_itr = author_date[0].children
        author = 'authorHtmlCss'
        while 'authorHtmlCss' in author:
            author = next(author_itr).text
    except:
        author = ''
    date = author_date[1].text

    # process data
    date = re.findall('[0-3][0-9]\.[0-1][0-9]\.[0-2][0-9]',date)[0]
    date = datetime.strptime(date, '%d.%m.%y').date()

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
