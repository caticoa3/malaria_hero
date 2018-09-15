#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 21:30:22 2017

@author: carlos
"""
import httplib2
import html5lib
from bs4 import BeautifulSoup

def grab_links(url):
    'Extracts the urls of links on a webpage.' 
    'The substring_list is used to delete any undesirable links that were extracted.'

    http = httplib2.Http()
    status, response = http.request(url)
    soup = BeautifulSoup(response, 'html.parser')
            
    urls_on_page = []
    for link in soup.find_all('a'):
        urls_on_page.append(link.get('href'))
    #print(set(urls_on_page))
    
    urls_on_page = set(urls_on_page)
    urls_on_page = list(urls_on_page)
    
    substring_list = ['tel', 'intechnic', 'redirect','chrome.google.com','www.google.com']
    
    rmv_list = []
    for x in urls_on_page:
        if any(substring in str(x) for substring in substring_list):
            rmv_list.append(str(x))
        elif 'http' not in str(x):
            rmv_list.append(str(x))
        elif type(x)==None:
            rmv_list.append(x)
    
#    print(rmv_list)
    for y in rmv_list:
        try:
            urls_on_page.remove(str(y))  
        except ValueError:
            del urls_on_page[urls_on_page.index(None)]
    #        urls_on_page = list(filter(None, urls_on_page))
    #        print(urls_on_page)
    return urls_on_page

test = grab_links('https://www.intechnic.com/blog/60-beautiful-examples-of-one-page-website-design-inspirations/')
                  
#'https://www.similarweb.com/top-websites/category/health/products-and-shopping'


