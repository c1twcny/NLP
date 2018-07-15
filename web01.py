import urllib3
import urllib
import requests
from bs4 import BeautifulSoup
from lxml import html
from lxml.html import fromstring
import re


link_list = []
link_list_r = []

link = 'https://www.wsj.com/'
keywords = ['wsj', 'barron']


with urllib.request.urlopen('https://www.wsj.com') as f:
    wsj0 = f.read().decode('utf-8')
soup = BeautifulSoup(wsj0, 'html.parser')
#print(soup.prettify())

for l in soup.find_all('a'):
    wsj_link = l.get('href')
    link_list.append(wsj_link)

link_list = [l for l in link_list if len(str(l)) >=8]

pattern = re.compile('^http[s]?\:.*wsj\.[a-z] | ^http[s]?\:.*barron\.[a-z]')

tmp_string = str(link_list[100])
#print(str(tmp_string))
#result = re.search('^http[s]?\:.*wsj\.[a-z]*', str(tmp_string))
#print(result)

for l in link_list:
    tmp_string = str(l)
    result = re.search('http[s]?\:.*wsj\.[a-z]*', tmp_string)
    if result != None:
        link_list_r.append(l)


for l in link_list_r:
    print(l)

print(len(link_list_r))