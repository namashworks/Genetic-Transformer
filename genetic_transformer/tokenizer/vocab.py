"""Vocabulary class for token-index mapping."""


PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
SEP_IDX = 4


class Vocabulary:
    """Maps tokens to integer indices and back."""

    SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>", "<SEP>"]

    def __init__(self):
        self.token2idx: dict[str, int] = {}
        self.idx2token: dict[int, str] = {}
        self._frozen = False

    def build_from_tokens(self, token_list: list[str]) -> None:
        """Build vocabulary from special tokens + provided token list."""
        self.token2idx = {}
        self.idx2token = {}
        # Add special tokens first
        for token in self.SPECIAL_TOKENS:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token
        # Add provided tokens
        for token in token_list:
            if token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token
        self._frozen = True

    def encode(self, token: str) -> int:
        """Convert a single token to its index."""
        return self.token2idx.get(token, UNK_IDX)

    def decode(self, idx: int) -> str:
        """Convert an index back to its token."""
        return self.idx2token.get(idx, "<UNK>")

    def __len__(self) -> int:
        return len(self.token2idx)

    def __contains__(self, token: str) -> bool:
        return token in self.token2idx
