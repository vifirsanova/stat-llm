#!/usr/bin/env python3
import requests
import re
from bs4 import BeautifulSoup

def get_response(url):
  """
  Get response from URL
  args: 
    - url (str)
  return: soup object
  """
  headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
  }

  response = requests.get(url, headers=headers)
  if not response.raise_for_status():
    return BeautifulSoup(response.text, "lxml")

def filter_links(links):
  """
  (1) Filter URLs with '_',' - ', '/' and page parameters
  (2) Filter URLs containing one-letter words
  return: links that do not contain words with vowels
  """
  # Filter URLs with '_',' - ', '/' and page parameters
  pattern = re.compile(
      r'https://ru\.wiktionary\.org/wiki/[^/]*([?&_\-]|/)'
  )
  clean_links_1 = {link for link in links if not pattern.search(link)}
  # Filter URLs containing one-letter words
  pattern = re.compile(
    r'https://ru\.wiktionary\.org/wiki/[^/]{1}?$'
  )
  clean_links_2 = {link for link in clean_links_1 if not pattern.search(link)}
  # Filter links that do not contain words with vowels
  final_links = []
  # Iterate over the link list
  for link in clean_links_2:
    # If link contain vowels, append
    if bool(re.search(r'[аоеуыюиэёя]', link)):
      final_links.append(link)
      
  return final_links

def get_wiki_links(links):
  """
  Get links with words from wiki-page
  return: list of relevant links
  """
  # Get links with words from wiktionary
  word_links = [get_response(link).find_all('a', rel="mw:WikiLink") for link in links]

  # Create set of links with words
  words = set()

  for word_link in word_links:
    for link in word_link:
      if 'Категория' not in link.get('href') and 'Индекс' not in link.get('href'):
        if link.get('class') is None:
          words.add("https:" + link.get('href'))
        if link.get('class') is not None and 'mw-selflink-fragment' not in link.get('class'):
          words.add("https:" + link.get('href'))
  return words

def get_morph(link):
  """
  Find all word formations from tables in wiktionary article
  args:
    - link to parse data
  return: set of word forms from a given page
  """
  # Find all table rows in a page
  response = get_response(link).find_all('table', class_='morfotable ru')[0].find_all('td', bgcolor=None)
  wordforms = [re.sub(r'́', '', link.text) for link in response]
  # Create set of unique words
  wf = []
  for word in wordforms:
    if wordforms is not None:
      if len(word.split()) > 1:
        words = word.split()
        wf.extend(words)
      else:
        wf.append(word)
  if len(wf) > 0:
    return '\n'.join(list(set(wf)))

if __name__ == "__main__":
  # Get links from А to Я
  paragraphs = get_response("https://ru.wiktionary.org/wiki/Индекс:Русский_язык").find_all("a")
  links = ["https:" + p.get('href') for p in paragraphs[51:51+33]]
  print('Parsed single-letter data successfully!')
  # Get links with words from wiki-page
  words = get_wiki_links(links)
  print('Parsed wiki links successfully!')
  # Filter the list of links with words from '_',' - ', '/' and page parameters
  clean_links = filter_links(words)

  # Get letter sequences from 2 wiktionary pages 
  soup = get_response('https://ru.wiktionary.org/wiki/%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%92%D0%B8%D0%BA%D0%B8%D1%81%D0%BB%D0%BE%D0%B2%D0%B0%D1%80%D1%8C:%D0%98%D0%BD%D0%B4%D0%B5%D0%BA%D1%81:%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9_%D1%8F%D0%B7%D1%8B%D0%BA').find_all('a')
  soup.extend(get_response('https://ru.wiktionary.org/w/index.php?title=%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%92%D0%B8%D0%BA%D0%B8%D1%81%D0%BB%D0%BE%D0%B2%D0%B0%D1%80%D1%8C:%D0%98%D0%BD%D0%B4%D0%B5%D0%BA%D1%81:%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9_%D1%8F%D0%B7%D1%8B%D0%BA&pagefrom=%D0%A8%2F%D0%B5+%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9+%D1%8F%D0%B7%D1%8B%D0%BA%0A%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9+%D1%8F%D0%B7%D1%8B%D0%BA%2F%D0%A8%2F%D0%B5#mw-pages').find_all('a'))
  print('Parsed data from letter sequences successfully!')
  # Find links with pattern href:letter_a/letter_b
  links = []
  pattern = re.compile(r"^Индекс:Русский язык\/[^\/]+\/[^\/]+")
  for link in soup:
    if pattern.match(link.text):
      links.append("https://ru.wiktionary.org" + link.get('href'))
  print('Parsed wiki data successfully!')
  # Get links with words from wiki-page
  words = get_wiki_links(links)
  print('Parsed wiki links successfully!')
  # Filter the list of links with words from '_',' - ', '/' and page parameters
  clean_links.extend(filter_links(words))

  # Clean all duplicates from data
  clean_links = set(clean_links)
  print('Begin parsing morphology..')
  # Parse the data
  with open('morphs_output.txt', 'w', encoding='utf-8') as output_file:
    for link in list(clean_links):
      try:
        output_file.write(f'{get_morph(link)}\n')
        output_file.write('\n')
      except IndexError as e:
        pass
