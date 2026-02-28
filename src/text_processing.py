"""
Text processing utilities for the Nupe language TTS system.

Nupe (Nupencin) is a Nupoid language spoken in Niger State, Nigeria.  The
orthography uses the standard Latin alphabet extended with a small set of
additional characters to represent phonemes that are not present in English.
Tone is marked with diacritic symbols (acute ´ and grave `).

This module provides:
- A canonical character vocabulary for Nupe text.
- Functions to normalise, encode, and decode text sequences.
"""

import re
import unicodedata
from typing import List

# ---------------------------------------------------------------------------
# Character vocabulary
# ---------------------------------------------------------------------------

# Padding / special symbols
PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

# Nupe character set: lower-case Latin letters, Nupe-specific characters,
# digits, whitespace, and common punctuation.
_NUPE_CHARS: List[str] = list(
    "abcdefghijklmnopqrstuvwxyz"
    "ɛɔəɡ"          # IPA vowels and the labio-velar voiced stop used in Nupe
    "áéíóúàèìòùāēīōū"  # vowels with tonal / length diacritics
    "ɛ́ɛ̀ɔ́ɔ̀"          # diacritic forms of Nupe-specific vowels
    " ',-."
)

# Deduplicate while preserving order (a character may appear in multiple
# categories above if Unicode normalisation collapses them).
_seen: set = set()
_NUPE_CHARS_DEDUP: List[str] = []
for _c in _NUPE_CHARS:
    if _c not in _seen:
        _seen.add(_c)
        _NUPE_CHARS_DEDUP.append(_c)

VOCAB: List[str] = [PAD_TOKEN, EOS_TOKEN, UNK_TOKEN] + _NUPE_CHARS_DEDUP

CHAR2IDX = {ch: idx for idx, ch in enumerate(VOCAB)}
IDX2CHAR = {idx: ch for idx, ch in enumerate(VOCAB)}

PAD_IDX = CHAR2IDX[PAD_TOKEN]
EOS_IDX = CHAR2IDX[EOS_TOKEN]
UNK_IDX = CHAR2IDX[UNK_TOKEN]

VOCAB_SIZE = len(VOCAB)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalise a Nupe text string before encoding.

    Steps:
    1. Unicode NFC normalisation to compose diacritics.
    2. Lower-case.
    3. Collapse multiple whitespace characters into a single space.
    4. Strip leading / trailing whitespace.

    Parameters
    ----------
    text:
        Raw input string.

    Returns
    -------
    str
        Normalised string.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


# ---------------------------------------------------------------------------
# Encoding / decoding
# ---------------------------------------------------------------------------

def encode(text: str, add_eos: bool = True) -> List[int]:
    """Encode a normalised Nupe string into a sequence of vocabulary indices.

    Unknown characters are mapped to ``UNK_IDX``.  An EOS token is appended
    when *add_eos* is ``True`` (the default).

    Parameters
    ----------
    text:
        Input text (will be normalised internally).
    add_eos:
        Whether to append an EOS token at the end of the sequence.

    Returns
    -------
    List[int]
        Integer index sequence.
    """
    text = normalize_text(text)
    indices = [CHAR2IDX.get(ch, UNK_IDX) for ch in text]
    if add_eos:
        indices.append(EOS_IDX)
    return indices


def decode(indices: List[int], strip_special: bool = True) -> str:
    """Decode a sequence of vocabulary indices back to a string.

    Parameters
    ----------
    indices:
        Sequence of integer indices.
    strip_special:
        When ``True``, PAD, EOS, and UNK tokens are removed from the output.

    Returns
    -------
    str
        Decoded string.
    """
    special = {PAD_IDX, EOS_IDX, UNK_IDX} if strip_special else set()
    chars = [IDX2CHAR[i] for i in indices if i not in special and i in IDX2CHAR]
    return "".join(chars)
