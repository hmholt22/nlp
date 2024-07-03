import requests
from bs4 import BeautifulSoup
import re

def get_psalm_text(psalm):
    ps_text = ''
    flag = 0 #declare flag to get only text of the psalm

    for line in psalm:
        tmp_str = line.get_text() #Get text from each line

        if flag == 0:
            if tmp_str[0] == '1': flag = 1 #start concatenating lines starting with first verse

        tmp_str = tmp_str.replace(u'\xa0', u' ') #eliminate line returns
        tmp_str = ''.join([i for i in tmp_str if not i.isdigit()]) #eliminate verse numbers

        if flag == 1:
            ps_text += ' ' + tmp_str #concat string

        ps_text = re.sub(' +', ' ', ps_text) #remove additional white space

    return ps_text.strip()

def create_ps_dict(first_psalm = 1, last_psalm = 150, verbose = False):
#get all psalms
    psalm_dict = {}

    for i in range(first_psalm, last_psalm+1):
        web_address = 'https://www.biblegateway.com/passage/?search=Psalm+' + str(i) + '&version=NRSVCE'
        psalm = requests.get(web_address)
        soup = BeautifulSoup(psalm.content, 'html.parser')
        
        if verbose:
            title = soup.title.text
            print(title)

        #regexp to find all lines for all verses
        regex_str = "^text Ps-" + str(i) +"-.*?$"
        reg_exp = re.compile(regex_str)
        ps = soup.find_all('span', class_=reg_exp)

        ps_text = get_psalm_text(ps) #Clean html and get text for psalm
        
        #remove literary pauses
        ps_text = ps_text.replace('Selah', '') 
        ps_test = ps_text.replace('Higgaion', '')
        
        ps_text = re.sub(r'\[[^)]\]', '', ps_text) #remove footnote links

        psalm_dict[i] = ps_text
        
    return psalm_dict