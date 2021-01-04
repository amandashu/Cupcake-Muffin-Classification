import os
import csv
import pandas as pd
import pickle
import time
import requests

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common import exceptions

import unicodedata
import re

def sba_get_links(driver, in_link):
    """
    Inputs link and returns list of recipe links to scrape for SBA
    """
    driver.get(in_link)
    time.sleep(10)
    recipe_links = []

    i = 1
    while True:
        print(i)
        i+=1

        time.sleep(5)
        html = driver.page_source
        soup = BeautifulSoup(html, "html5lib")
        body = soup.find('div', {"class":"c-content-panel"})
        recipe_links +=  [a['href'] for a in body.find_all('a', href=True)]

        try:
            next_page = WebDriverWait(driver, 15).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//*[contains(text(), 'Next Page >>')]")
                    ))
            driver.execute_script("arguments[0].click();", next_page)
        except exceptions.TimeoutException as e:
            break

    return recipe_links

def sba_get_data(driver, in_link):
    """
    Scrapes a recipe link and returns dictionary of data for SBA
    """
    driver.get(in_link)
    time.sleep(10)

    html = driver.page_source
    soup = BeautifulSoup(html, "html5lib")
    ingredients_body = soup.find('div',{"class":"tasty-recipes-ingredients"}).find('ul')
    ingredients = [unicodedata.normalize('NFKD',x.text).replace("*","") for x in ingredients_body.find_all('li')]

    out_dct = {'link': in_link, 'ingredients':ingredients}
    return out_dct

def bb_get_links(driver, in_link):
    """
    Inputs link and returns list of recipe links to scrape for BB
    """
    driver.get(in_link)
    time.sleep(7)

    for i in range(5):
        print(i)
        time.sleep(5)
        try:
            driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
            load_more_button = WebDriverWait(driver, 20).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//*[contains(text(), 'Load more')]")
                    )).click()
        except exceptions.TimeoutException as e:
            break

    html = driver.page_source
    soup = BeautifulSoup(html, "html5lib")
    recipe_links =  [a['href'] for a in soup.find_all('a',{"class":"button"}, href=True)]
    return recipe_links

def bb_get_data(driver, in_link):
    """
    Scrapes a recipe link and returns dictionary of data for BB
    """
    driver.get(in_link)
    time.sleep(7)
    html = driver.page_source
    soup = BeautifulSoup(html, "html5lib")

    body =  soup.find('div', {"class":"isopad"})
    ingredients = body.find('p', attrs={'style': 'text-align: center;'}).text.split('\n')[1:]
    out_dct = {'link': in_link, 'ingredients':ingredients}
    return out_dct

def bc_get_links(driver, in_link):
    """
    Inputs link and returns list of recipe links to scrape for BC
    """
    driver.get(in_link)
    time.sleep(7)
    html = driver.page_source
    soup = BeautifulSoup(html, "html5lib")

    recipe_links = []

    for i in range(5):
        body = soup.find('main', {"class":"content"})
        recipe_links +=  [a['href'] for a in body.find_all('a', {"class":"entry-title-link"}, href=True)]
        print(recipe_links)

        next = soup.find('a', text='Next Page »')
        if next:
            driver.get(next['href'])
            time.sleep(7)
            html = driver.page_source
            soup = BeautifulSoup(html, "html5lib")
        else:
            break
    return recipe_links


def bc_get_data(driver, in_link):
    """
    Scrapes a recipe link and returns dictionary of data for BC
    """
    driver.get(in_link)
    time.sleep(7)
    html = driver.page_source
    soup = BeautifulSoup(html, "html5lib")

    body = soup.find('div',{"class":"ERSIngredients"})
    if body is None:
        body = soup.find('div',{"class":"mv-create-ingredients"})

    if body.find('div',text=re.compile(r'([M|m]uffin)|([C|c]upcake)')):
        body = body.find('div',text=re.compile(r'(Muffin)|(Cupcake)')).find_next('ul')
    else:
        body = body.find('ul')

    ingredients = [x.text.replace("*","").strip() for x in body.find_all('li') if x.text != '\n']
    out_dct = {'link': in_link, 'ingredients':ingredients}
    return out_dct

def scrape_data(chromedriver_path):
    """
    Writes data to csv
    """
    start_links = {
                    'sba_muffins': 'https://sallysbakingaddiction.com/recipes/?fwp_breakfast_category=muffins',
                    'sba_cupcakes': 'https://sallysbakingaddiction.com/recipes/?fwp_desserts_category=cupcakes',
                    'bb_muffins': 'https://bakingbites.com/category/recipes/muffins/',
                    'bb_cupcakes': 'https://bakingbites.com/category/recipes/cupcakes/',
                    'bc_muffins': 'https://www.thebakerchick.com/category/recipes/muffins/',
                    'bc_cupcakes': 'https://www.thebakerchick.com/category/recipes/cupcakes/'
                    }

    func_dct = {
                'sba': [sba_get_links, sba_get_data],
                'bb': [bb_get_links, bb_get_data],
                'bc': [bc_get_links, bc_get_data]
                }

    driver = webdriver.Chrome(chromedriver_path)
    csv_path = 'data/recipes.csv'

    dirname = 'data'
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    for item, val in start_links.items():
        pickle_path = 'data/' + item + '.pickle'
        web_abb = item[:item.find('_')]
        if not os.path.isfile(pickle_path):
            recipe_links = func_dct[web_abb][0](driver, val)
            with open(pickle_path , 'wb') as f:
                pickle.dump(recipe_links, f)

    all_pickles = [dirname +  '/' + filename for filename in os.listdir(dirname) if filename.endswith('pickle')]

    for p in all_pickles:
        web_abb = p[p.find('/')+1:p.find('_')]

        with open(p, 'rb') as file:
            links = pickle.load(file)

        #write data
        headers = ['link', 'type', 'ingredients']

        if not os.path.isfile(csv_path):
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, headers)
                writer.writerow({x:x for x in headers})

        if 'muffin' in p:
            type = 'muffin'
        elif 'cupcake' in p:
            type = 'cupcake'

        with open(csv_path, 'a', newline='', encoding='UTF-8') as f:
            writer = csv.DictWriter(f, headers)

            #go through each link and write data to csv
            for l in links:
                print(l)
                try:
                    data_dct = func_dct[web_abb][1](driver, l)
                except Exception as e:
                    print(e)
                    data_dct = {'link': l, 'ingredients': None}

                data_dct['type'] = type
                writer.writerow(data_dct)
                print(data_dct)
    return
