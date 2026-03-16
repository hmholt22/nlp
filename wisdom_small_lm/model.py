import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Shared base — the embedding layer every architecture has in common
# ---------------------------------------------------------------------------

class _BaseModel(nn.Module):
    """
    All three models inherit this so the token embedding lives in one place.
    This is the pedagogical core: regardless of how sequences are processed
    (RNN, LSTM, or Transformer), the first step is always the same lookup table
    mapping character indices → dense vectors.
    """
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Architecture 1: RNN
# ---------------------------------------------------------------------------

class RNNModel(_BaseModel):
    """
    Embedding → single-direction RNN → linear projection to vocab logits.
    Simplest possible sequence model — good baseline and easiest to inspect.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden_dim: int = 256, n_layers: int = 2, dropout: float = 0.1):
        super().__init__(vocab_size, embed_dim)
        # dropout only applies between layers, so disable it for single-layer models
        self.rnn = nn.RNN(
            embed_dim, hidden_dim, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0
        )
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        embed = self.embedding(x)       # (B, T, embed_dim)
        hidden, _ = self.rnn(embed)     # (B, T, hidden_dim)
        return self.output(hidden)      # (B, T, vocab_size)


# ---------------------------------------------------------------------------
# Architecture 2: LSTM
# ---------------------------------------------------------------------------

class LSTMModel(_BaseModel):
    """
    Embedding → LSTM → linear projection.
    Adds a cell state (long-term memory gate) on top of the RNN hidden state —
    better at capturing dependencies across longer sequences.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden_dim: int = 256, n_layers: int = 2, dropout: float = 0.1):
        super().__init__(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0
        )
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        embed = self.embedding(x)       # (B, T, embed_dim)
        hidden, _ = self.lstm(embed)    # (B, T, hidden_dim)
        return self.output(hidden)      # (B, T, vocab_size)


# ---------------------------------------------------------------------------
# Architecture 3: Transformer
# ---------------------------------------------------------------------------

class TransformerModel(_BaseModel):
    """
    Embedding + positional embedding → causal TransformerEncoder → linear projection.

    Two embeddings:
      - Token embedding (shared with RNN/LSTM via _BaseModel): what character is this?
      - Positional embedding (learned): where in the sequence is this character?

    The causal mask ensures position i can only attend to positions <= i,
    which is required for next-character prediction (no peeking ahead).

    Note: embed_dim must be divisible by n_heads.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 64, n_heads: int = 4,
                 n_layers: int = 2, ff_dim: int = 256, context_len: int = 128,
                 dropout: float = 0.1):
        super().__init__(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(context_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output = nn.Linear(embed_dim, vocab_size)
        self.context_len = context_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        B, T = x.shape
        positions = torch.arange(T, device=x.device)               # (T,)
        embed = self.embedding(x) + self.pos_embedding(positions)   # (B, T, embed_dim)

        # Upper-triangle mask: True = "ignore this key position"
        # → query i cannot attend to any key j > i (causal / autoregressive)
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device), diagonal=1
        ).bool()

        out = self.transformer(embed, mask=causal_mask)  # (B, T, embed_dim)
        return self.output(out)                          # (B, T, vocab_size)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(arch: str = "transformer", vocab_size: int = 67,
                embed_dim: int = 64, hidden_dim: int = 256,
                n_heads: int = 4, n_layers: int = 2, ff_dim: int = 256,
                context_len: int = 128, dropout: float = 0.1) -> _BaseModel:
    """
    Return a model ready for training.

    Args:
        arch:        "transformer", "lstm", or "rnn"
        vocab_size:  Number of unique characters — get this from len(char_to_idx)
        embed_dim:   Token embedding dimension (shared by all architectures)
        hidden_dim:  RNN/LSTM hidden state size (ignored by Transformer)
        n_heads:     Attention heads (Transformer only; embed_dim must be divisible by n_heads)
        n_layers:    Number of stacked layers
        ff_dim:      Feedforward dim inside each Transformer block (ignored by RNN/LSTM)
        context_len: Max sequence length — must match the value used in dataset.py
        dropout:     Dropout rate applied during training
    """
    arch = arch.lower()
    if arch == "rnn":
        return RNNModel(vocab_size, embed_dim, hidden_dim, n_layers, dropout)
    elif arch == "lstm":
        return LSTMModel(vocab_size, embed_dim, hidden_dim, n_layers, dropout)
    elif arch == "transformer":
        return TransformerModel(vocab_size, embed_dim, n_heads, n_layers,
                                ff_dim, context_len, dropout)
    else:
        raise ValueError(f"Unknown arch '{arch}'. Choose from: 'rnn', 'lstm', 'transformer'")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters — useful for comparing architectures."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)