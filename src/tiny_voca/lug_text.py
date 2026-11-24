"""Luganda text clean-up before the spelling rules in lug.py.

Numbers, money, percentages, clock times, phone numbers, a few titles,
common loanwords and initialisms are written out as Luganda words, which the
rules in lug.py then read like any other word. ``expand`` runs inside
``tiny_voca.text.phonemize`` for the "lug" front end.

None of this is in the front-end fingerprint (``lug.identity``): its output is
ordinary Luganda spelling, so the released weights still load and nothing has
to be retrained. ``web/public/lug.js`` has the same code (``expand``); keep the
two alike, tests/test_lug.py compares them.

Numbers are said in the counting form (emu, bbiri, ssatu ...), which is also
the form that agrees with nouns of class 9/10 (ente, essaawa, eddakiika). A
number after a noun of another class is not made to agree: "abantu 3" is read
"abantu ssatu", not "abantu basatu". The tables below want a native speaker's
check; changing them needs no retraining.
"""

import re

UNITS = ("", "emu", "bbiri", "ssatu", "nnya", "ttaano", "mukaaga", "musanvu", "munaana", "mwenda")
TENS = ("", "kkumi", "amakumi abiri", "amakumi asatu", "amakumi ana", "amakumi ataano",
        "nkaaga", "nsanvu", "kinaana", "kyenda")
HUNDREDS = ("", "kikumi", "bibiri", "bisatu", "bina", "bitaano", "lukaaga", "lusanvu", "lunaana", "lwenda")
THOUSANDS = ("", "lukumi", "enkumi bbiri", "enkumi ssatu", "enkumi nnya", "enkumi ttaano",
             "kakaaga", "kasanvu", "kanaana", "kenda")
DIGITS = ("zeero",) + UNITS[1:]
# 1-5 agree with the noun counted; 6 and up do not change.
CLASS_4 = ("", "gumu", "ebiri", "esatu", "ena", "etaano")          # emitwalo: tens of thousands
CLASS_14 = ("", "bumu", "bubiri", "busatu", "buna", "butaano")      # obukadde, obuwumbi
JOIN = " mu "
POINT = "pointi"
LARGEST = 10 ** 12                                                  # from here on, digit by digit

# Titles written with a full stop. The stop is dropped with the title, so it does not end a sentence.
TITLES = {"dr": "dokita", "prof": "pulofeesa", "owek": "owekitiibwa"}
# Words the Luganda rules would misread, respelled the way they are said in Luganda. Whole words, any case.
RESPELLINGS = {
    "ok": "okee", "okay": "okee", "dr": "dokita", "doctor": "dokita", "prof": "pulofeesa",
    "professor": "pulofeesa", "owek": "owekitiibwa", "president": "pulezidenti", "minister": "minisita",
    "police": "poliisi", "school": "sukulu", "bus": "bbaasi", "taxi": "takisi", "phone": "fooni",
    "computer": "kompyuta", "internet": "yintaneti", "email": "imeeri", "radio": "leediyo",
    "tv": "ttivvi", "television": "ttivvi", "covid": "kovidi", "whatsapp": "wasaapu",
    "facebook": "feesibuuku", "youtube": "yuutuubu",
}
# Letter names, for initialisms such as NRM or KCCA.
LETTER_NAMES = {
    "a": "ee", "b": "bbi", "c": "ssi", "d": "ddi", "e": "ii", "f": "efu", "g": "jji", "h": "eeci", "i": "aayi",
    "j": "jjee", "k": "kke", "l": "eru", "m": "emu", "n": "enu", "o": "oo", "p": "pi", "q": "kyu", "r": "aa",
    "s": "esi", "t": "tti", "u": "yu", "v": "vi", "w": "dabuliyu", "x": "eksi", "y": "wayi", "z": "zeddi",
}

_NUMBER = r"[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?"
_TITLE = re.compile(r"\b(dr|prof|owek)\.", re.I | re.A)
_TIME = re.compile(r"\b(?:e?ssaawa\s+)?([0-9]{1,2}):([0-9]{2})\b", re.I | re.A)
_MONEY_BEFORE = re.compile(r"\b(?:ugx|ushs?|shs?)\.?\s?(" + _NUMBER + r")(?:\s?/[=-])?", re.I | re.A)
_MONEY_AFTER = re.compile(r"(" + _NUMBER + r")\s?(?:/[=-]|(?:ugx|shs?)\b)", re.I | re.A)
_DOLLARS = re.compile(r"\$\s?(" + _NUMBER + r")", re.A)
_PERCENT = re.compile(r"(" + _NUMBER + r")\s?%", re.A)
# "+256 ..." or a leading zero ("0772 ..."), not the zeros inside 1,000 or 3.05.
_PHONE = re.compile(r"(?:\+|(?<![0-9A-Za-z.,])0(?=[0-9]))[0-9]+(?:[ -][0-9]+)*", re.A)
_PLAIN = re.compile(_NUMBER, re.A)
_WORD = re.compile(r"(?<![A-Za-z0-9'À-ɏ])[A-Za-z]+(?![A-Za-z0-9'À-ɏ])")
_SPACES = re.compile(" {2,}")


def digits(text):
    """Every digit of ``text`` on its own: "0772" -> "zeero musanvu musanvu bbiri"."""
    return " ".join(DIGITS[int(char)] for char in text if char in "0123456789")


def _unit(unit, forms):
    return forms[unit] if unit < len(forms) else UNITS[unit]


def _below_hundred(n, forms):
    tens, unit = divmod(n, 10)
    if not tens:
        return _unit(unit, forms)
    if not unit:
        return TENS[tens]
    word = _unit(unit, forms)
    if tens == 1:                                                   # kkumi n'emu, kkumi na bbiri
        return "kkumi " + ("n'" + word if word[0] in "aeiou" else "na " + word)
    return TENS[tens] + JOIN + word


def _below_thousand(n, forms=UNITS):
    hundreds, rest = divmod(n, 100)
    parts = [HUNDREDS[hundreds]] if hundreds else []
    if rest:
        parts.append(_below_hundred(rest, forms))
    return JOIN.join(parts)


def _thousands(n):
    """1-999 thousands: lukumi, enkumi bbiri ... kenda, then emitwalo (tens of thousands)."""
    if n < 10:
        return THOUSANDS[n]
    tens, rest = divmod(n, 10)
    word = "omutwalo" if tens == 1 else "emitwalo " + _below_thousand(tens, CLASS_4)
    return word + (JOIN + THOUSANDS[rest] if rest else "")


def _scale(n, one, many):
    return one if n == 1 else many + " " + _below_thousand(n, CLASS_14)


def number(n):
    """A whole number in Luganda words, counting form: 2026 -> "enkumi bbiri mu amakumi abiri mu mukaaga"."""
    if n == 0:
        return DIGITS[0]
    if n >= LARGEST:
        return digits(str(n))
    parts = []
    for size, say in ((10 ** 9, lambda k: _scale(k, "akawumbi", "obuwumbi")),
                      (10 ** 6, lambda k: _scale(k, "akakadde", "obukadde")),
                      (10 ** 3, _thousands)):
        count, n = divmod(n, size)
        if count:
            parts.append(say(count))
    if n:
        parts.append(_below_thousand(n))
    return JOIN.join(parts)


def amount(text):
    """A written number ("1,500", "3.25", "007") in words. Digits after the point are read one by one."""
    whole, _, fraction = text.replace(",", "").partition(".")
    said = digits(whole) if len(whole) > 1 and whole[0] == "0" else number(int(whole))
    return said + (f" {POINT} " + digits(fraction) if fraction else "")


def clock(hour, minute, said_in_luganda):
    """"essaawa ..." for a clock time. Luganda counts the hours from 7 o'clock (essaawa emu), so a time in
    international digits (13:30) moves by six hours; one written after "essaawa" is taken as Luganda time."""
    if not said_in_luganda:
        hour += 6
    words = "essaawa " + number(hour % 12 or 12)
    if minute == 30:
        return words + " n'ekitundu"
    return words + (" n'eddakiika " + number(minute) if minute else "")


def expand(text):
    """``text`` with numbers, signs, titles, loanwords and initialisms written out in Luganda words."""
    shouting = not re.search(r"[a-z]", text)                       # a text all in capitals has no initialisms

    def time(match):
        hour, minute = int(match[1]), int(match[2])
        if hour > 23 or minute > 59:
            return match[0]
        return clock(hour, minute, match[0][0] not in "0123456789")

    def word(match):
        said = match[0]
        if said.lower() in RESPELLINGS:
            return RESPELLINGS[said.lower()]
        if not shouting and 2 <= len(said) <= 5 and said.isupper():
            return " ".join(LETTER_NAMES[letter] for letter in said.lower())
        return said

    text = _TITLE.sub(lambda match: TITLES[match[1].lower()], text)
    text = _TIME.sub(time, text)
    text = _MONEY_BEFORE.sub(lambda match: "siringi " + amount(match[1]), text)
    text = _MONEY_AFTER.sub(lambda match: "siringi " + amount(match[1]), text)
    text = _DOLLARS.sub(lambda match: "doola " + amount(match[1]), text)
    text = _PERCENT.sub(lambda match: "ebitundu " + amount(match[1]) + " ku kikumi", text)
    text = _PHONE.sub(lambda match: digits(match[0]), text)
    text = _PLAIN.sub(lambda match: amount(match[0]), text)
    text = text.replace("&", " ne ")
    text = _WORD.sub(word, text)
    return _SPACES.sub(" ", text).strip()
