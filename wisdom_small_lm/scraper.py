import requests
from bs4 import BeautifulSoup
import re
import json
import time


# Book configuration: search term (for URL), CSS abbreviation, and chapter count
BOOKS = {
    "psalms":        {"search": "Psalm",         "abbrev": "Ps",   "chapters": 150},
    "job":           {"search": "Job",            "abbrev": "Job",  "chapters": 42},
    "proverbs":      {"search": "Proverbs",       "abbrev": "Prov", "chapters": 31},
    "ecclesiastes":  {"search": "Ecclesiastes",   "abbrev": "Eccl", "chapters": 12},
    "song_of_songs": {"search": "Song of Songs",  "abbrev": "Song", "chapters": 8},
    "wisdom":        {"search": "Wisdom",         "abbrev": "Wis",  "chapters": 19},
    "sirach":        {"search": "Sirach",         "abbrev": "Sir",  "chapters": 51},
}


def get_verses(book_key, chapter):
    """Scrape all verses from a single chapter of a book."""
    book = BOOKS[book_key]
    search = book["search"].replace(" ", "+")
    abbrev = book["abbrev"]

    url = f"https://www.biblegateway.com/passage/?search={search}+{chapter}&version=NRSVCE"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    pattern = re.compile(rf"^text {abbrev}-\d+-\d+$")
    spans = soup.find_all("span", class_=pattern)

    verse_dict = {}
    for span in spans:
        # Skip standalone heading spans — they have an id but no sup child
        # e.g. <span id="..." class="text Song-4-1">The Bride's Beauty Extolled</span>
        # Actual verse-start spans always contain either a versenum sup (v2+)
        # or a chapternum span (v1), never neither
        if span.get('id') and not span.find('sup') and not span.find(class_='chapternum'):
            continue

        text = span.get_text().strip()
        if not text or text.isdigit() or text == "Selah":
            continue

        verse_class = next((c for c in span["class"] if re.match(rf"^{abbrev}-\d+-\d+$", c)), None)
        if not verse_class:
            continue

        verse_num = int(verse_class.split("-")[-1])
        text = text.replace('\xa0', ' ').strip()
        text = re.sub(r'\[[^\]]*\]', '', text)
        text = re.sub(r' +', ' ', text).strip()

        if verse_num not in verse_dict:
            # New verse start — covers both explicit versenum sup and verse 1
            # which uses a chapter number display instead of a versenum sup
            text = re.sub(r'^\d+\s*', '', text)  # strip leading chapter/verse number
            verse_dict[verse_num] = text
        else:
            verse_dict[verse_num] += " " + text

    return [{"verse_number": num, "text": text} for num, text in sorted(verse_dict.items())]


def scrape_book(book_key, verbose=False, delay=0.5):
    """Scrape all chapters of a book. Returns dict keyed by chapter number."""
    book = BOOKS[book_key]
    book_data = {}

    for chapter in range(1, book["chapters"] + 1):
        if verbose:
            print(f"  Scraping {book['search']} {chapter}/{book['chapters']}...")
        verses = get_verses(book_key, chapter)
        book_data[chapter] = [
            v for v in verses
            if len(v["text"]) > 20 and not v["text"].startswith("(")
        ]
        time.sleep(delay)

    return book_data


def scrape_all_books(book_keys=None, verbose=False, delay=0.5):
    """Scrape multiple books. Defaults to all books in BOOKS."""
    if book_keys is None:
        book_keys = list(BOOKS.keys())

    corpus = {}
    for book_key in book_keys:
        if verbose:
            print(f"\n=== Scraping {book_key.replace('_', ' ').title()} ===")
        corpus[book_key] = scrape_book(book_key, verbose=verbose, delay=delay)

    return corpus


def save_corpus(corpus, filename='wisdom_corpus.json'):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=4)
    print(f"Saved corpus to {filename}")


def load_corpus(filename='wisdom_corpus.json'):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)
