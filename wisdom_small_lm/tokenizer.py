import json
import numpy as np
from pathlib import Path

CORPUS_PATH = Path(__file__).parent / "wisdom_corpus.json"
VOCAB_PATH = Path(__file__).parent / "vocab.json"
ENCODED_PATH = Path(__file__).parent / "encoded_corpus.npy"


def load_corpus(path=CORPUS_PATH):
    """Flatten all verse texts from the corpus JSON into a single string."""
    with open(path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    verses = []
    for book, chapters in corpus.items():
        for chapter, verse_list in chapters.items():
            for verse in verse_list:
                verses.append(verse["text"])

    return "\n".join(verses)


def build_vocab(text):
    """Return char→index and index→char mappings for all unique characters in text."""
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char


def encode(text, char_to_idx):
    """Encode a string as a flat numpy array of integers."""
    # int16 is sufficient — vocab will be well under 32,767 characters
    return np.array([char_to_idx[ch] for ch in text], dtype=np.int16)


def save_vocab(char_to_idx, idx_to_char, path=VOCAB_PATH):
    """Save both vocab mappings to JSON. idx_to_char keys are stored as strings (JSON requirement)."""
    vocab = {
        "char_to_idx": char_to_idx,
        "idx_to_char": {str(k): v for k, v in idx_to_char.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(path=VOCAB_PATH):
    """Load vocab mappings from JSON. Restores idx_to_char keys to integers."""
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    char_to_idx = vocab["char_to_idx"]
    idx_to_char = {int(k): v for k, v in vocab["idx_to_char"].items()}
    return char_to_idx, idx_to_char


def tokenize_and_save():
    print("Loading corpus...")
    text = load_corpus()
    print(f"  Total characters : {len(text):,}")
    print(f"  Total verses     : {text.count(chr(10)) + 1:,}")

    print("\nBuilding vocabulary...")
    char_to_idx, idx_to_char = build_vocab(text)
    vocab_size = len(char_to_idx)
    print(f"  Vocab size: {vocab_size}")
    print(f"  Characters: {repr(''.join(sorted(char_to_idx.keys())))}")

    print("\nEncoding corpus...")
    encoded = encode(text, char_to_idx)
    print(f"  Encoded shape : {encoded.shape}")
    print(f"  dtype         : {encoded.dtype}")

    print(f"\nSaving vocab  → {VOCAB_PATH}")
    save_vocab(char_to_idx, idx_to_char)

    print(f"Saving encoded → {ENCODED_PATH}")
    np.save(ENCODED_PATH, encoded)

    print("\nDone.")
    return text, char_to_idx, idx_to_char, encoded


if __name__ == "__main__":
    tokenize_and_save()