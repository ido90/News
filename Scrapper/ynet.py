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
with ~400 articles ordered by sections.
'''

def get_homepage(url="https://www.ynet.co.il"):
    return st.get_homepage(url)

def get_all_sections(home, verbose=1):
    # get sections
    sections = st.get_all_links_with_strings(home, ('href="/home/',
                                                    '"bananasDataLayerRprt') )
    urls = [s.get('href') for s in sections]
    titles = [s.get('onclick')[len("bananasDataLayerRprt('"):-2]
              for s in sections]
    # complete relative urls
    urls = st.relative_to_absolute_url(urls, r"https://www.ynet.co.il")
    # filter undesired sections
    desired_sections = ('חדשות','כלכלה','ספורט','תרבות',
                        'דיגיטל','בריאות וכושר','צרכנות',
                        'נדל"ן','חופש','אוכל','מדע','יחסים',
                        'דעות','קריירה')
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
    articles = st.get_all_links_with_strings(source, ('/articles/',), True)
    if demo:
        articles = articles[:3]
    urls = [a.get('href') for a in articles]
    urls = st.relative_to_absolute_url(urls, r"https://www.ynet.co.il")
    titles = [a.text for a in articles]
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
        sheet_name='ynet_demo' if demo else 'ynet',
        update_freq=update_freq, verbose=verbose
    )

def get_article_data(url):
    # get page
    try:
        html = urlopen(url)
    except:
        warn(f'Bad URL: {url:s}')
        return 'BAD_URL'
    soup = BeautifulSoup(html, 'lxml')
    # verify not ynet+ (i.e. limited) article
    if soup.find_all(class_='premium_image')\
            or soup.find_all(class_='premuim_image'):
        return 'IRRELEVANT_PAGE'
    # get data from page
    try:
        if '.calcalist.co.il' in url:
            title = soup.find_all(class_='art-title')[0].text
            subtitle = soup.find_all(class_='art-sub-title')[0].text
            author = soup.find_all(class_='art-author')[0].text
            date = soup.find_all(class_='l-date')[1].text
            text = '\n'.join([par.text for par in soup.find_all('p')
                              if len(par.text)>30 and not '\t\t\t' in par.text])
        elif 'pplus.ynet.co.il' in url:
            pass
        else:
            title = soup.find_all(class_='art_header_title')[0].text
            subtitle = soup.find_all(class_='art_header_sub_title')
            subtitle = subtitle[0].text if subtitle else None
            author_date = soup.find_all(class_='art_header_footer_author')
            try:
                author_itr = author_date[0].children
                author = 'authorHtmlCss'
                while 'authorHtmlCss' in author:
                    author = next(author_itr).text
            except StopIteration:
                author = ''
            date = author_date[1].text
            text = '\n'.join([par.text for par in soup.find_all('p')
                              if len(par.text)>30 and not '\t\t\t' in par.text])
    except:
        print(url)
        raise

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
                           save_to=r'..\Data\articles')
    print('Done.')
