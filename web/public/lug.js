// The Luganda front end in plain JavaScript: text -> tiny-voca phoneme ids.
//
// A copy of the Python path `tiny_voca.text.phonemize` with the "lug" front end:
// `normalize_text` and `chunks` from text.py, `expand` from lug_text.py (numbers
// and the like in words), the spelling rules of lug.py and `tokenize`. The model only understands the ids that Python would give, so the
// two must stay alike; tests/test_lug.py compares them sentence by sentence.
//
// Loaded with importScripts() in the worker, or with require() in node (tests).

const lug = (() => {
  // tiny_voca.text.SYMBOLS. Append-only: the ids are the positions in this list.
  const SPECIALS = ["<pad>", "<bos>", "<eos>", "<sp>", "<unk>"];
  const [, BOS, EOS, SP, UNK] = SPECIALS;
  const PUNCTUATION = [",", ".", "?", "!", ";", ":", '"', "'", "(", ")", "—", "…"];
  const CONSONANTS = ["p", "b", "t", "d", "k", "ɡ", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",
                      "m", "n", "ŋ", "l", "ɹ", "j", "w", "tʃ", "dʒ", "ɾ", "ʔ"];
  const VOWELS = ["ə", "ɐ", "ɚ", "ᵻ", "ɪ", "i", "iː", "ɛ", "æ", "ʌ", "ɑː", "ɔ", "ɔː", "ʊ", "uː", "ɜː",
                  "eɪ", "aɪ", "ɔɪ", "oʊ", "aʊ", "iə", "oː",
                  "ɑːɹ", "ɔːɹ", "oːɹ", "ɪɹ", "ɛɹ", "ʊɹ", "aɪɚ", "aɪə", "əl", "n̩"];
  const PHONES = new Set([...CONSONANTS, ...VOWELS]);
  const SYMBOLS = [...SPECIALS, ...PUNCTUATION, ...CONSONANTS, ...VOWELS];
  const ID = new Map(SYMBOLS.map((symbol, index) => [symbol, index]));
  const MARKS = new Set(PUNCTUATION);

  // text.py: _QUOTES, ABBREVIATIONS, normalize_text, chunks.
  const QUOTES = { "‘": "'", "’": "'", "‚": "'", "‛": "'", "`": "'", "´": "'",
                   "“": '"', "”": '"', "„": '"', "‟": '"', "«": '"', "»": '"',
                   "–": "—", "―": "—", "[": "(", "{": "(", "]": ")", "}": ")" };
  const ABBREVIATIONS = new Set(["mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc"]);
  const isAlnum = (c) => /^[\p{L}\p{N}]$/u.test(c);
  const isAlpha = (c) => /^\p{L}$/u.test(c);
  const isDigit = (c) => /^\p{Nd}$/u.test(c);

  function normalizeText(text) {
    text = Array.from(text.normalize("NFKC"), (c) => QUOTES[c] ?? c).join("");
    text = text.replace(/\.{3,}/g, "…");
    text = text.replace(/-{2,}|\s-\s/g, "—");
    return text.replace(/\s+/g, " ").trim();
  }

  // [isPunctuation, piece] pairs. Same cuts as text.chunks.
  function chunks(text) {
    const chars = Array.from(text);
    const pieces = [];
    let start = 0;
    for (let i = 0; i < chars.length; i++) {
      const char = chars[i];
      if (!MARKS.has(char)) continue;
      const previous = i ? chars[i - 1] : "";
      const following = i + 1 < chars.length ? chars[i + 1] : "";
      if (char === "'" && isAlnum(previous) && isAlnum(following)) continue;
      if (".,:".includes(char) && isDigit(previous) && isDigit(following)) continue;
      if (char === ".") {
        const word = chars.slice(0, i).join("").match(/([A-Za-z]+)$/);
        if (word && ABBREVIATIONS.has(word[1].toLowerCase())) continue;
        if (isAlpha(previous) && isAlpha(following)) continue;
      }
      const piece = chars.slice(start, i).join("").trim();
      if (piece) pieces.push([false, piece]);
      pieces.push([true, char]);
      start = i + 1;
    }
    const rest = chars.slice(start).join("").trim();
    if (rest) pieces.push([false, rest]);
    return pieces;
  }

  // lug.py: VOWELS, DIGRAPHS, LETTERS, clean, word_phones, words.
  const LUG_VOWELS = new Map([["a", ["ɐ", "ɑː"]], ["e", ["ɛ", "eɪ"]], ["i", ["i", "iː"]], ["o", ["ɔ", "oː"]], ["u", ["ʊ", "uː"]]]);
  const DIGRAPHS = new Map([["ng'", ["ŋ"]], ["ny", ["n", "j"]], ["ky", ["tʃ"]], ["gy", ["dʒ"]], ["sh", ["ʃ"]], ["ch", ["tʃ"]]]);
  const LETTERS = new Map([["b", ["b"]], ["c", ["tʃ"]], ["d", ["d"]], ["f", ["f"]], ["g", ["ɡ"]], ["h", ["h"]], ["j", ["dʒ"]],
                           ["k", ["k"]], ["l", ["l"]], ["m", ["m"]], ["n", ["n"]], ["p", ["p"]], ["q", ["k"]], ["r", ["ɾ"]],
                           ["s", ["s"]], ["t", ["t"]], ["v", ["v"]], ["w", ["w"]], ["x", ["k", "s"]], ["y", ["j"]], ["z", ["z"]],
                           ["ŋ", ["ŋ"]]]);

  function clean(text) {
    text = text.normalize("NFC").toLowerCase().replaceAll("’", "'").replaceAll("ʼ", "'").replaceAll("‘", "'");
    // Strip accents (é -> e) but keep ŋ, which has no decomposition.
    text = Array.from(text.normalize("NFD")).filter((c) => !/\p{Mn}/u.test(c)).join("");
    return text.replaceAll("ng'", "ŋ");
  }

  function wordPhones(word) {
    const phones = [];
    let i = 0;
    while (i < word.length) {
      const char = word[i];
      if (char === "'") { i += 1; continue; }
      if (LUG_VOWELS.has(char)) {
        const long = i + 1 < word.length && word[i + 1] === char;
        phones.push(LUG_VOWELS.get(char)[long ? 1 : 0]);
        i += long ? 2 : 1;
        continue;
      }
      const pair = word.slice(i, i + 2);
      // Geminate digraph: kky, ggy, nny -> the digraph twice.
      const next = word.slice(i + 1, i + 3);
      if (i + 2 < word.length && word[i + 1] === char && DIGRAPHS.has(next)) {
        phones.push(...(next !== "ny" ? [...DIGRAPHS.get(next), ...DIGRAPHS.get(next)] : ["n", "n", "j"]));
        i += 3;
        continue;
      }
      if (DIGRAPHS.has(pair)) {
        phones.push(...DIGRAPHS.get(pair));
        i += 2;
        continue;
      }
      if (LETTERS.has(char)) phones.push(...LETTERS.get(char));
      i += 1; // anything else (digits, symbols) is skipped
    }
    return phones;
  }

  function words(piece) {
    return (clean(piece).match(/[a-zŋ']+/g) || []).map(wordPhones).filter((phones) => phones.length);
  }

  // lug_text.py: numbers, money, times, titles, loanwords and initialisms written out in Luganda words before
  // the rules read them. Not part of the checkpoint's fingerprint. Keep the tables and steps as in Python.
  const UNITS = ["", "emu", "bbiri", "ssatu", "nnya", "ttaano", "mukaaga", "musanvu", "munaana", "mwenda"];
  const TENS = ["", "kkumi", "amakumi abiri", "amakumi asatu", "amakumi ana", "amakumi ataano",
                "nkaaga", "nsanvu", "kinaana", "kyenda"];
  const HUNDREDS = ["", "kikumi", "bibiri", "bisatu", "bina", "bitaano", "lukaaga", "lusanvu", "lunaana", "lwenda"];
  const THOUSANDS = ["", "lukumi", "enkumi bbiri", "enkumi ssatu", "enkumi nnya", "enkumi ttaano",
                     "kakaaga", "kasanvu", "kanaana", "kenda"];
  const DIGITS = ["zeero", ...UNITS.slice(1)];
  const CLASS_4 = ["", "gumu", "ebiri", "esatu", "ena", "etaano"];
  const CLASS_14 = ["", "bumu", "bubiri", "busatu", "buna", "butaano"];
  const JOIN = " mu ";
  const POINT = "pointi";
  const LARGEST = 10 ** 12;
  const TITLES = { dr: "dokita", prof: "pulofeesa", owek: "owekitiibwa" };
  const RESPELLINGS = {
    ok: "okee", okay: "okee", dr: "dokita", doctor: "dokita", prof: "pulofeesa",
    professor: "pulofeesa", owek: "owekitiibwa", president: "pulezidenti", minister: "minisita",
    police: "poliisi", school: "sukulu", bus: "bbaasi", taxi: "takisi", phone: "fooni",
    computer: "kompyuta", internet: "yintaneti", email: "imeeri", radio: "leediyo",
    tv: "ttivvi", television: "ttivvi", covid: "kovidi", whatsapp: "wasaapu",
    facebook: "feesibuuku", youtube: "yuutuubu",
  };
  const LETTER_NAMES = {
    a: "ee", b: "bbi", c: "ssi", d: "ddi", e: "ii", f: "efu", g: "jji", h: "eeci", i: "aayi",
    j: "jjee", k: "kke", l: "eru", m: "emu", n: "enu", o: "oo", p: "pi", q: "kyu", r: "aa",
    s: "esi", t: "tti", u: "yu", v: "vi", w: "dabuliyu", x: "eksi", y: "wayi", z: "zeddi",
  };
  const NUMBER = "[0-9]{1,3}(?:,[0-9]{3})+(?:\\.[0-9]+)?|[0-9]+(?:\\.[0-9]+)?";
  const TITLE = /\b(dr|prof|owek)\./gi;
  const TIME = /\b(?:e?ssaawa\s+)?([0-9]{1,2}):([0-9]{2})\b/gi;
  const MONEY_BEFORE = new RegExp("\\b(?:ugx|ushs?|shs?)\\.?\\s?(" + NUMBER + ")(?:\\s?/[=-])?", "gi");
  const MONEY_AFTER = new RegExp("(" + NUMBER + ")\\s?(?:/[=-]|(?:ugx|shs?)\\b)", "gi");
  const DOLLARS = new RegExp("\\$\\s?(" + NUMBER + ")", "g");
  const PERCENT = new RegExp("(" + NUMBER + ")\\s?%", "g");
  const PHONE = /(?:\+|(?<![0-9A-Za-z.,])0(?=[0-9]))[0-9]+(?:[ -][0-9]+)*/g;
  const PLAIN = new RegExp(NUMBER, "g");
  const WORD = /(?<![A-Za-z0-9'À-ɏ])[A-Za-z]+(?![A-Za-z0-9'À-ɏ])/g;

  const digits = (text) => Array.from(text).filter((c) => "0123456789".includes(c)).map((c) => DIGITS[+c]).join(" ");
  const unit = (n, forms) => (n < forms.length ? forms[n] : UNITS[n]);

  function belowHundred(n, forms) {
    const tens = Math.floor(n / 10), one = n % 10;
    if (!tens) return unit(one, forms);
    if (!one) return TENS[tens];
    const word = unit(one, forms);
    if (tens === 1) return "kkumi " + ("aeiou".includes(word[0]) ? "n'" + word : "na " + word);
    return TENS[tens] + JOIN + word;
  }

  function belowThousand(n, forms = UNITS) {
    const hundreds = Math.floor(n / 100), rest = n % 100;
    const parts = hundreds ? [HUNDREDS[hundreds]] : [];
    if (rest) parts.push(belowHundred(rest, forms));
    return parts.join(JOIN);
  }

  function thousands(n) {
    if (n < 10) return THOUSANDS[n];
    const tens = Math.floor(n / 10), rest = n % 10;
    const word = tens === 1 ? "omutwalo" : "emitwalo " + belowThousand(tens, CLASS_4);
    return word + (rest ? JOIN + THOUSANDS[rest] : "");
  }

  const scale = (n, one, many) => (n === 1 ? one : many + " " + belowThousand(n, CLASS_14));

  function number(n) {
    if (n === 0) return DIGITS[0];
    if (n >= LARGEST) return digits(String(n));
    const parts = [];
    for (const [size, say] of [[10 ** 9, (k) => scale(k, "akawumbi", "obuwumbi")],
                               [10 ** 6, (k) => scale(k, "akakadde", "obukadde")],
                               [10 ** 3, thousands]]) {
      const count = Math.floor(n / size);
      n %= size;
      if (count) parts.push(say(count));
    }
    if (n) parts.push(belowThousand(n));
    return parts.join(JOIN);
  }

  function amount(text) {
    const [whole, ...rest] = text.replaceAll(",", "").split(".");
    const fraction = rest.join(".");
    // More than 12 digits: digit by digit, as number() does from LARGEST on (and +whole would lose digits).
    const said = (whole.length > 1 && whole[0] === "0") || whole.length > 12 ? digits(whole) : number(+whole);
    return said + (fraction ? ` ${POINT} ` + digits(fraction) : "");
  }

  function clock(hour, minute, saidInLuganda) {
    if (!saidInLuganda) hour += 6;
    const words = "essaawa " + number(hour % 12 || 12);
    if (minute === 30) return words + " n'ekitundu";
    return words + (minute ? " n'eddakiika " + number(minute) : "");
  }

  function expand(text) {
    const shouting = !/[a-z]/.test(text);
    text = text.replace(TITLE, (_, title) => TITLES[title.toLowerCase()]);
    text = text.replace(TIME, (all, hour, minute) => (+hour > 23 || +minute > 59 ? all
                                                      : clock(+hour, +minute, !"0123456789".includes(all[0]))));
    text = text.replace(MONEY_BEFORE, (_, value) => "siringi " + amount(value));
    text = text.replace(MONEY_AFTER, (_, value) => "siringi " + amount(value));
    text = text.replace(DOLLARS, (_, value) => "doola " + amount(value));
    text = text.replace(PERCENT, (_, value) => "ebitundu " + amount(value) + " ku kikumi");
    text = text.replace(PHONE, (all) => digits(all));
    text = text.replace(PLAIN, (all) => amount(all));
    text = text.replaceAll("&", " ne ");
    text = text.replace(WORD, (word) => {
      const key = word.toLowerCase();
      if (Object.hasOwn(RESPELLINGS, key)) return RESPELLINGS[key];
      if (!shouting && word.length >= 2 && word.length <= 5 && /^[A-Z]+$/.test(word))
        return Array.from(key, (letter) => LETTER_NAMES[letter]).join(" ");
      return word;
    });
    return text.replace(/ {2,}/g, " ").trim();
  }

  // text.tokenize, for words that carry no stress. null when there is nothing to pronounce.
  function phonemize(text) {
    const symbols = [BOS];
    for (const [punctuation, piece] of chunks(expand(normalizeText(text)))) {
      if (punctuation) {
        if (symbols[symbols.length - 1] !== piece) symbols.push(piece); // collapse repeats such as "!!"
        continue;
      }
      for (const word of words(piece)) {
        const last = symbols[symbols.length - 1];
        if (PHONES.has(last) || last === UNK) symbols.push(SP);
        for (const phone of word) symbols.push(PHONES.has(phone) ? phone : UNK);
      }
    }
    symbols.push(EOS);
    if (!symbols.some((symbol) => PHONES.has(symbol) || symbol === UNK)) return null;
    return { ids: symbols.map((symbol) => ID.get(symbol)), stress: symbols.map(() => 0), text: symbols.join(" ") };
  }

  return { SYMBOLS, normalizeText, chunks, clean, wordPhones, words, number, expand, phonemize };
})();

if (typeof module !== "undefined") module.exports = lug;
