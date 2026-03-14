import requests
from bs4 import BeautifulSoup
import re
import json


def get_psalm_verses(psalm):
    url = f"https://www.biblegateway.com/passage/?search=Psalm+{psalm}&version=NRSVCE"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    psalm_text = soup.find_all("span", class_=re.compile(r"^text Ps-\d+-\d+$"))

    verse_dict = {}
    for span in psalm_text:
        text = span.get_text().strip()
        if text and not text.isdigit() and text != "Selah":
            verse_class = next((c for c in span["class"] if re.match(r"Ps-\d+-\d+", c)), None)
            if not verse_class:
                continue
            verse_num = int(verse_class.split("-")[-1])
            text = text.replace(u'\xa0', ' ').strip()
            text = re.sub(r'\[[^]]*\]', '', text)
            text = re.sub(r' +', ' ', text)
            if span.find("sup", class_="versenum"):
                verse_dict[verse_num] = text
            elif verse_num in verse_dict:
                verse_dict[verse_num] += " " + text

    for span in psalm_text:
        text = span.get_text().strip()
        if text and not text.isdigit() and text != "Selah":
            verse_class = next((c for c in span["class"] if re.match(r"Ps-\d+-\d+", c)), None)
            if not verse_class:
                continue
            verse_num = int(verse_class.split("-")[-1])
            text = text.replace(u'\xa0', ' ').strip()
            text = re.sub(r'\[[^]]*\]', '', text)
            text = re.sub(r' +', ' ', text)
            if span.find("sup", class_="versenum"):
                text = re.sub(r'^\d+\s', '', text)
                verse_dict[verse_num] = text
            elif verse_num in verse_dict:
                verse_dict[verse_num] += " " + text

    return [{"verse_number": num, "text": text} for num, text in verse_dict.items()]


def create_ps_dict_verses(first_psalm=1, last_psalm=150, verbose=False):
    ps_dict = {}
    for psalm in range(first_psalm, last_psalm + 1):
        if verbose:
            print(f"Scraping Psalm {psalm}")
        verses = get_psalm_verses(psalm)
        ps_dict[psalm] = [
            v for v in verses
            if not any(kw in v["text"].upper() for kw in ["BOOK", "PSALM", "–", "PSALMS"])
            and len(v["text"]) > 20
            and not v["text"].startswith("(")
        ]
    return ps_dict


def load_ps_dict(filename='ps_verses.json'):
    """Load psalm dictionary from JSON file."""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_ps_dict(ps_dict, filename='ps_verses.json'):
    """Save psalm dictionary to JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(ps_dict, f, ensure_ascii=False, indent=4)
