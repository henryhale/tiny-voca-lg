"""Luganda front end: spelling -> tiny-voca inventory phones.

Luganda spelling is close to phonemic, so plain rules are enough:
* a double vowel is a long vowel (aa -> ɑː, ee -> eɪ, ii -> iː, oo -> oː, uu -> uː);
* a double consonant is a geminate and is spoken as the phone twice (kk -> k k);
* ny is the palatal nasal (n j), ng' (or ŋ) is the velar nasal, ky/gy are palatal stops;
* other apostrophes mark elision (n'omwana) and are dropped.

Every output phone is already in the English inventory, so the English weights
and the phoneme aligner can be reused unchanged. Tone is not written in Luganda
spelling and is not modelled. Digits are not read: spell numbers out first.

The tables below are part of the checkpoint's front-end fingerprint
(``identity``): changing any of them, or VERSION, means the released
tiny-voca-lg weights no longer load. ``web/public/lug.js`` is the same rules in
JavaScript for the browser demo; keep the two alike.
"""

import re
import unicodedata

VERSION = "lug-rules-1"

VOWELS = {"a": ("ɐ", "ɑː"), "e": ("ɛ", "eɪ"), "i": ("i", "iː"), "o": ("ɔ", "oː"), "u": ("ʊ", "uː")}
DIGRAPHS = {"ng'": ("ŋ",), "ny": ("n", "j"), "ky": ("tʃ",), "gy": ("dʒ",), "sh": ("ʃ",), "ch": ("tʃ",)}
LETTERS = {"b": ("b",), "c": ("tʃ",), "d": ("d",), "f": ("f",), "g": ("ɡ",), "h": ("h",), "j": ("dʒ",),
           "k": ("k",), "l": ("l",), "m": ("m",), "n": ("n",), "p": ("p",), "q": ("k",), "r": ("ɾ",),
           "s": ("s",), "t": ("t",), "v": ("v",), "w": ("w",), "x": ("k", "s"), "y": ("j",), "z": ("z",),
           "ŋ": ("ŋ",)}
_WORD = re.compile(r"[a-zŋ']+")


def identity():
    """Everything that decides the output, for the front-end fingerprint."""
    return {"version": VERSION, "vowels": VOWELS, "digraphs": DIGRAPHS, "letters": LETTERS}


def clean(text):
    text = unicodedata.normalize("NFC", text).lower().replace("’", "'").replace("ʼ", "'").replace("‘", "'")
    # Strip accents (é -> e) but keep ŋ, which has no decomposition.
    text = "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))
    return text.replace("ng'", "ŋ")


def word_phones(word):
    phones, i = [], 0
    while i < len(word):
        char = word[i]
        if char == "'":
            i += 1
            continue
        if char in VOWELS:
            long = i + 1 < len(word) and word[i + 1] == char
            phones.append(VOWELS[char][1 if long else 0])
            i += 2 if long else 1
            continue
        pair = word[i:i + 2]
        # Geminate digraph: kky, ggy, nny -> the digraph twice.
        if i + 2 < len(word) and word[i + 1] == char and word[i + 1:i + 3] in DIGRAPHS:
            phones.extend(DIGRAPHS[word[i + 1:i + 3]] * 2 if word[i + 1:i + 3] != "ny" else ("n", "n", "j"))
            i += 3
            continue
        if pair in DIGRAPHS:
            phones.extend(DIGRAPHS[pair])
            i += 2
            continue
        if char in LETTERS:
            phones.extend(LETTERS[char])
        i += 1  # anything else (digits, symbols) is skipped
    return phones


def words(piece):
    """One text piece (no sentence punctuation) -> words as lists of (phone, stress)."""
    result = []
    for word in _WORD.findall(clean(piece)):
        phones = word_phones(word)
        if phones:
            result.append([(phone, 0) for phone in phones])
    return result