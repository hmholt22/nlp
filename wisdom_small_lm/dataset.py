import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from tokenizer import ENCODED_PATH


class WisdomDataset(Dataset):
    """
    Sliding-window character-level dataset.

    Each sample is a pair (x, y) of integer sequences of length context_len.
    y is x shifted right by one position — i.e. y[i] is the character the
    model should predict after seeing x[0:i+1].  This is standard next-character
    prediction used to train language models.

    Example with context_len=5:
        encoded = [4, 7, 2, 9, 1, 3, 8]
        sample 0 → x=[4,7,2,9,1]  y=[7,2,9,1,3]
        sample 1 → x=[7,2,9,1,3]  y=[2,9,1,3,8]
    """

    def __init__(self, encoded: np.ndarray, context_len: int = 128):
        # Store as int64 — PyTorch embedding layers require LongTensor
        self.data = torch.from_numpy(encoded.astype(np.int64))
        self.context_len = context_len

    def __len__(self):
        # Every starting position up to (total - context_len) yields a valid sample
        return len(self.data) - self.context_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.context_len]
        y = self.data[idx + 1 : idx + self.context_len + 1]
        return x, y


def load_datasets(context_len: int = 128, val_split: float = 0.1,
                  encoded_path: Path = ENCODED_PATH):
    """
    Load the encoded corpus and return (train_dataset, val_dataset).

    The split is positional (first 90% / last 10%) rather than shuffled —
    shuffling would leak future context into training, which defeats the
    purpose of a language model.

    Args:
        context_len:  Number of characters per input window. Start with 128;
                      increase to 256+ to let the model learn across verses.
        val_split:    Fraction of the corpus reserved for validation.
        encoded_path: Path to the .npy file produced by tokenizer.py.
    """
    encoded = np.load(encoded_path)
    split_idx = int(len(encoded) * (1 - val_split))

    train_ds = WisdomDataset(encoded[:split_idx], context_len=context_len)
    val_ds   = WisdomDataset(encoded[split_idx:],  context_len=context_len)

    print(f"context_len  : {context_len}")
    print(f"train samples: {len(train_ds):,}")
    print(f"val samples  : {len(val_ds):,}")

    return train_ds, val_ds