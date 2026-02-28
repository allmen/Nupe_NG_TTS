"""
Unit tests for src.text_processing.
"""

import pytest
from src.text_processing import (
    VOCAB,
    VOCAB_SIZE,
    CHAR2IDX,
    IDX2CHAR,
    PAD_IDX,
    EOS_IDX,
    UNK_IDX,
    PAD_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
    normalize_text,
    encode,
    decode,
)


class TestVocabulary:
    def test_special_tokens_at_start(self):
        assert VOCAB[PAD_IDX] == PAD_TOKEN
        assert VOCAB[EOS_IDX] == EOS_TOKEN
        assert VOCAB[UNK_IDX] == UNK_TOKEN

    def test_char2idx_idx2char_roundtrip(self):
        for idx, ch in IDX2CHAR.items():
            assert CHAR2IDX[ch] == idx

    def test_vocab_size_matches(self):
        assert VOCAB_SIZE == len(VOCAB)

    def test_no_duplicates(self):
        assert len(VOCAB) == len(set(VOCAB))

    def test_ascii_letters_present(self):
        for ch in "abcdefghijklmnopqrstuvwxyz":
            assert ch in CHAR2IDX, f"'{ch}' missing from vocab"


class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("Emi") == "emi"

    def test_collapse_whitespace(self):
        assert normalize_text("emi  shi") == "emi shi"

    def test_strip_whitespace(self):
        assert normalize_text("  emi ") == "emi"

    def test_nfc_normalisation(self):
        # Decomposed form (NFD) should be composed to NFC.
        import unicodedata

        nfd = unicodedata.normalize("NFD", "ó")
        result = normalize_text(nfd)
        assert result == unicodedata.normalize("NFC", "ó")


class TestEncode:
    def test_returns_list_of_ints(self):
        indices = encode("emi")
        assert isinstance(indices, list)
        assert all(isinstance(i, int) for i in indices)

    def test_eos_appended_by_default(self):
        indices = encode("emi")
        assert indices[-1] == EOS_IDX

    def test_no_eos_when_disabled(self):
        indices = encode("emi", add_eos=False)
        assert EOS_IDX not in indices

    def test_unknown_char_maps_to_unk(self):
        indices = encode("emi😀", add_eos=False)
        assert UNK_IDX in indices

    def test_known_chars_correct_indices(self):
        indices = encode("ab", add_eos=False)
        assert indices == [CHAR2IDX["a"], CHAR2IDX["b"]]

    def test_empty_string(self):
        indices = encode("", add_eos=True)
        assert indices == [EOS_IDX]


class TestDecode:
    def test_roundtrip(self):
        original = "emi shi"
        indices = encode(original, add_eos=False)
        recovered = decode(indices)
        assert recovered == original

    def test_strip_special_tokens(self):
        indices = [PAD_IDX, CHAR2IDX["a"], EOS_IDX]
        result = decode(indices, strip_special=True)
        assert result == "a"

    def test_keep_special_tokens(self):
        indices = [EOS_IDX, CHAR2IDX["a"]]
        result = decode(indices, strip_special=False)
        assert EOS_TOKEN in result

    def test_empty_sequence(self):
        assert decode([]) == ""
