import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import time
import io
from pprint import pprint
from warnings import warn
from datetime import datetime
from urllib.request import urlopen
from bs4 import BeautifulSoup
from openpyxl import load_workbook

'''
A bunch of general tools useful for articles scrapping from news website.

Note: a more scalable design could be a class of the news website,
containing classes of news sections, containing a list of urls.
This way, all the url-depending functions below wouldn't require passing
the whole list every call.
This is not a problem in our case of several calls and hundreds of url strings.

Note: due to much of iterative development over more and more websites with
many slight parsing differences, the code for the various websites still
suffers from the fundamental sin of significant code duplication.
'''

def get_homepage(url):
    html = urlopen(url)
    soup = BeautifulSoup(html, 'lxml')
    # Note: for non-windows, 'lxml' might need to be replaced ('html'?)
    return soup

def remove_duplications(ll):
    duplications = 0
    valid_ids = []
    already_exist = set()
    for i,l in enumerate(ll):
        if l in already_exist:
            duplications += 1
            ll[i] = None
        else:
            valid_ids.append(i)
            already_exist.add(l)
    ll = [x for x in ll if x is not None]
    return (ll, tuple(valid_ids), duplications)

def get_all_links_with_strings(soup, strs, require_text=False,
                               str_requirement_fun=all):
    if require_text:
        return [x for x in soup.find_all('a')
               if str_requirement_fun([s in str(x) for s in strs])
                and x.text.strip()]
    else:
        return [x for x in soup.find_all('a')
                if str_requirement_fun([s in str(x) for s in strs])]

def relative_to_absolute_url(urls, base):
    return [url if url.startswith(('http://','https://')) else base + url
            for url in urls]

def print_scrapped_articles_summary(
        urls, titles=None, verbose=1,
        unique=True, header=None, duplications=None):
    if verbose >= 1:
        if header: print(f"Section: {header:s}")
        print(f"Number of{' (unique)' if unique else '':s} articles: ",
              f"{len(urls):d}", sep='')
        if duplications: print(f"Article Duplications: {duplications:d}")
        if verbose < 2:
            if titles:
                pprint(titles[:3])
                if len(urls) >= 15:
                    pprint(titles[-3:])
            pprint(urls[:3])
            if len(urls) >= 15:
                pprint(urls[-3:])
    if verbose >= 2:
        pprint(titles)
        pprint(urls)

def get_articles_data_from_sections(
        sections, get_data_fun, save_to=None,
        sheet_name='news', update_freq=10, verbose=2):
    '''
    :param sections: a dictionary of the form
           {section_name : (urls_list, urls_titles_list)}.
    '''
    df = pd.concat([
        get_articles_data(
            sections[sec][0], get_data_fun=get_data_fun,
            section=sec, urls_titles=sections[sec][1],
            update_freq=update_freq, verbose=verbose
        ) for sec in sections
    ])
    if save_to:
        save_data_frame(df, save_to, sheet_name)
    return df

def save_data_frame(df, path, sheet_name='sheet'):
    if path[-5:] != '.xlsx': path += '.xlsx'
    try:
        curr = load_workbook(path)
    except FileNotFoundError:
        curr = None
    writer = pd.ExcelWriter(path, engine='openpyxl')
    if curr: writer.book = curr
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    writer.save()
    writer.close()

def load_data_frame(path, sheets=None, verbose=1):
    if path[-5:] != '.xlsx': path += '.xlsx'
    if sheets is None:
        i = 0
        df = pd.read_excel(path,i)
        while True:
            i += 1
            try:
                df = pd.concat([df, pd.read_excel(path, i)])
            except IndexError:
                break
        if verbose >= 1:
            print(f'Sheets read: {i:d}')
    else:
        df = pd.read_excel(path,sheets[0])
        df['source'] = sheets[0]
        for s in sheets[1:]:
            df2 = pd.read_excel(path, s)
            df2['source'] = s
            df = pd.concat([df, df2])
    return df

def get_articles_data(urls, get_data_fun,
                      section=None, urls_titles=None, save_to=None,
                      update_freq=10, verbose=2):
    '''
    :param get_data_fun: a function that gets a single article's url,
           and returns a dict (with keys url,title,subtitle,author,date,text)
           or an error message (BAD_URL or IRRELEVANT_PAGE).
    :return: pandas data frame of articles data.
    '''

    # initialization
    cols = ('url', 'title', 'author', 'date', 'subtitle', 'text')
    data = {col: [] for col in cols}
    if urls_titles:
        data['link_title'] = []
    if section:
        data['section'] = []
    unavailable_urls = 0
    if section and update_freq > 0:
        print(f"\n{section:s}:")

    # scrap articles
    for i, url in enumerate(urls):
        # status update
        if verbose >= 2 and i % update_freq == 0:
            print(f'Successful scraps: {i-unavailable_urls:d}/',
                  f'{i:d}/{len(urls):d}', sep='')
        # call parser
        a = get_data_fun(url)
        # handle errors (could also be implemented using exceptions)
        if a in ('BAD_URL','IRRELEVANT_PAGE'):
            unavailable_urls += 1
            continue
        # update data
        for col in cols:
            data[col].append(a[col])
        if urls_titles:
            data['link_title'].append(urls_titles[i])
        if section:
            data['section'].append(section)

    if verbose >= 1:
        print(f'Successful scraps:',
              f'{len(urls)-unavailable_urls:d}/{len(urls):d}')

    # convert to data frame
    df_cols = (['section'] if section else []) + \
              (['link_title'] if urls_titles else []) + \
              list(cols) # cols are defined explicitly to ensure their order
    df = pd.DataFrame(data=data, columns=df_cols)
    if save_to:
        if save_to[-4:] != '.csv': save_to += '.csv'
        df.to_csv(save_to, header=True, index=False)

    return df

def soup2file(path, url=None, html=None, soup=None):
    if soup is None:
        if html is None:
            html = urlopen(url)
        soup = BeautifulSoup(html,'lxml')
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(str(soup))

def url2html(url, attempts=3,
             error_on_failure=True, verbose=0, convert_to_none=False):
    soup = None
    for i in range(attempts):
        soup = BeautifulSoup(urlopen(url), 'lxml')
        if not str(soup).startswith(r'<html><head><meta charset'):
            break
    if error_on_failure and str(soup).startswith(r'<html><head><meta charset'):
        raise ValueError
    if verbose >= 2:
        print(f'Attempts required to open {url:s}: {i+1:d}')
    if verbose >= 1 and str(soup).startswith(r'<html><head><meta charset'):
        warn(f'Could not get valid URL: {url:s}')
    if convert_to_none and str(soup).startswith(r'<html><head><meta charset'):
        soup = None
    return soup
