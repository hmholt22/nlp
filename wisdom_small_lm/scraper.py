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
    "lamentations":  {"search": "Lamentations",   "abbrev": "Lam",  "chapters": 5},
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
        versenum_sup = span.find("sup", class_="versenum")
        chapternum_span = span.find(class_="chapternum")

        if book_key != "psalms":
            # Skip standalone heading spans — they have an id but no sup and no chapternum.
            # e.g. <span id="..." class="text Song-4-1">The Bride's Beauty Extolled</span>
            if span.get('id') and not versenum_sup and not chapternum_span:
                continue

        verse_class = next((c for c in span["class"] if re.match(rf"^{abbrev}-\d+-\d+$", c)), None)
        if not verse_class:
            continue
        verse_num = int(verse_class.split("-")[-1])

        if versenum_sup:
            # Get text after the versenum sup only — this strips any section heading
            # that precedes the sup within the same span (e.g. "In Praise of Wisdom 1 All wisdom...")
            parts = []
            for node in versenum_sup.next_siblings:
                parts.append(node.get_text() if hasattr(node, 'get_text') else str(node))
            text = ''.join(parts).strip()
        else:
            text = span.get_text().strip()

        if not text or text.isdigit() or text == "Selah":
            continue

        text = text.replace('\xa0', ' ').strip()
        text = re.sub(r'\[[^\]]*\]', '', text)
        text = re.sub(r' +', ' ', text).strip()

        if book_key == "psalms":
            # Psalms verse 1 is often a superscription heading embedded in the chapternum
            # span — only start a new verse entry on an explicit versenum sup (verse 2+).
            # Mirrors the logic from the original psalm_scraper.py.
            if versenum_sup:
                verse_dict[verse_num] = text
            elif verse_num in verse_dict:
                verse_dict[verse_num] += " " + text
            # else: no versenum and verse not yet started → superscription, skip it
        else:
            if verse_num not in verse_dict:
                if not versenum_sup:
                    # Verse 1 via chapternum — strip up to two leading digit groups.
                    # Some books produce "ch# v# text" (e.g. Sirach 6 → "6 1 and do not...").
                    text = re.sub(r'^(\d+\s+){1,2}', '', text)
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
        if book_key == "psalms":
            book_data[chapter] = [
                v for v in verses
                if not any(kw in v["text"].upper() for kw in ["BOOK", "PSALM", "–", "PSALMS"])
                and len(v["text"]) > 20
                and not v["text"].startswith("(")
            ]
        else:
            book_data[chapter] = [
                v for v in verses
                if len(v["text"]) > 20 and not v["text"].startswith("(")
            ]

        # Sirach chapter 1: the prologue paragraphs have no verse markers in the HTML,
        # so they get merged into verse 1. Split them out here:
        # the chapternum "1" appears as a standalone " 1 " preceded by lowercase text,
        # marking where the prologue ends and the actual verse begins.
        if book_key == "sirach" and chapter == 1:
            new_verses = []
            for v in book_data[chapter]:
                if v["verse_number"] == 1:
                    match = re.search(r'(?<=[a-z.,;]) 1 (?=[A-Z])', v["text"])
                    if match:
                        prologue = v["text"][:match.start()].strip()
                        verse1 = v["text"][match.end():].strip()
                        if len(prologue) > 20:
                            new_verses.append({"verse_number": 0, "text": prologue})
                        new_verses.append({"verse_number": 1, "text": verse1})
                    else:
                        new_verses.append(v)
                else:
                    new_verses.append(v)
            book_data[chapter] = new_verses
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
