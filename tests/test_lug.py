"""The Luganda front end ("lug"), and its JavaScript copy for the browser demo.

The released tiny-voca-lg checkpoint was trained on exactly these tokens: the
expected outputs below were recorded before its training started.
"""

import json
from pathlib import Path
import shutil
import subprocess
import unittest

from tiny_voca import lug, lug_text, text
from tiny_voca.text import PHONE_SET, SYMBOLS, NothingToPronounce, phonemize

LUG_JS = Path(__file__).resolve().parents[1] / "web" / "public" / "lug.js"

SENTENCES = [
    "Oli otya? Nkwagala nnyo, ssebo.",
    "Ng'ombe za kitaawe n'abaana bajja.",
    "Abaana bagenda ku ssomero buli lunaku, naye leero baasigadde waka.",
    "Obummonde obuzungu buwooma nnyo naye bulumya enjala, olya n'obeera ng’atalidde.",
    "Kkyalo kino kirimu abantu bangi; ggwe ani?",
    "Webale nnyo okujja!!! Weeraba...",
    "Ŋŋaanya “ekitabo” — kye nnaasoma (enkya).",
    "Café é ò à: shamba ch x q.",
    "Bya 2026 ne 3.5",
    "Mu 'Kampala' abantu ba-ngi.",
    "Dr. Kato yasasula Shs 5,000 ku ssaawa 10:30, ebitundu 50% ku NRM & KCCA. Ok!",
    "Essimu ye 0772 123456, ne $3.50 ne 1,000,000 ne 2,500,000,000.",
]
NOTHING = ["こんにちは、世界。", "!!! ...", "\U0001F642"]

# Written out by lug_text before the rules read them. A native speaker should check these words.
NUMBERS = {
    0: "zeero", 1: "emu", 7: "musanvu", 10: "kkumi", 11: "kkumi n'emu", 12: "kkumi na bbiri",
    20: "amakumi abiri", 25: "amakumi abiri mu ttaano", 60: "nkaaga", 99: "kyenda mu mwenda",
    100: "kikumi", 105: "kikumi mu ttaano", 250: "bibiri mu amakumi ataano", 1000: "lukumi",
    2026: "enkumi bbiri mu amakumi abiri mu mukaaga", 6000: "kakaaga", 10000: "omutwalo",
    15000: "omutwalo mu enkumi ttaano", 50000: "emitwalo etaano",
    250000: "emitwalo amakumi abiri mu etaano", 1000000: "akakadde",
    3500000: "obukadde busatu mu emitwalo amakumi ataano", 2000000000: "obuwumbi bubiri",
    10 ** 12: " ".join(["emu"] + ["zeero"] * 12),
}
EXPANDED = {
    "1,000,000 ne 3.05": "akakadde ne ssatu pointi zeero ttaano",
    "Bya 2026 ne 3.5": "Bya enkumi bbiri mu amakumi abiri mu mukaaga ne ssatu pointi ttaano",
    "Shs 5,000": "siringi enkumi ttaano",
    "5,000/=": "siringi enkumi ttaano",
    "UGX 20,000": "siringi emitwalo ebiri",
    "$3.50": "doola ssatu pointi ttaano zeero",
    "50%": "ebitundu amakumi ataano ku kikumi",
    "10:30": "essaawa nnya n'ekitundu",                     # international time, six hours on
    "7:00": "essaawa emu",
    "essaawa 2:15": "essaawa bbiri n'eddakiika kkumi na ttaano",   # already Luganda time
    "0772 123456": "zeero musanvu musanvu bbiri emu bbiri ssatu nnya ttaano mukaaga",
    "Dr. Kato ne Owek. Mayiga": "dokita Kato ne owekitiibwa Mayiga",
    "NRM ne KCCA": "enu aa emu ne kke ssi ssi ee",
    "OLI OTYA": "OLI OTYA",                                    # all capitals: no initialisms
    "Ok, tukole": "okee, tukole",
    "Kato & Nakato": "Kato ne Nakato",
    "Webale nnyo": "Webale nnyo",
}


class LugandaFrontEndTests(unittest.TestCase):
    def setUp(self):
        self.previous = text.frontend()
        text.set_frontend("lug")

    def tearDown(self):
        text.set_frontend(self.previous)

    def test_the_tokens_the_model_was_trained_on(self):
        said = {s: " ".join(phonemize(s).symbols) for s in SENTENCES[:2]}
        self.assertEqual(said["Oli otya? Nkwagala nnyo, ssebo."],
                         "<bos> ɔ l i <sp> ɔ t j ɐ ? n k w ɐ ɡ ɐ l ɐ <sp> n n j ɔ , s s ɛ b ɔ . <eos>")
        self.assertEqual(said["Ng'ombe za kitaawe n'abaana bajja."],
                         "<bos> ŋ ɔ m b ɛ <sp> z ɐ <sp> k i t ɑː w ɛ <sp> n ɐ b ɑː n ɐ <sp> b ɐ dʒ dʒ ɐ . <eos>")

    def test_rules(self):
        self.assertEqual(lug.word_phones("kkyalo"), ["tʃ", "tʃ", "ɐ", "l", "ɔ"])      # geminate digraph
        self.assertEqual(lug.word_phones("nnyo"), ["n", "n", "j", "ɔ"])
        self.assertEqual(lug.word_phones("ŋŋaanya"), ["ŋ", "ŋ", "ɑː", "n", "j", "ɐ"])
        self.assertEqual(lug.word_phones("webale"), ["w", "ɛ", "b", "ɐ", "l", "ɛ"])
        self.assertEqual(lug.word_phones("2026"), [])                 # the rules skip digits; lug_text writes them out

    def test_every_phone_is_in_the_english_inventory(self):
        tables = [*lug.VOWELS.values(), *lug.DIGRAPHS.values(), *lug.LETTERS.values()]
        self.assertTrue(all(phone in PHONE_SET for table in tables for phone in table))
        for sentence in SENTENCES:
            self.assertEqual(phonemize(sentence).unknown, ())

    def test_identity(self):
        identity = text.frontend_identity()
        self.assertEqual((identity["frontend_version"], identity["phonemizer"]), ("lug-rules-1", "lug-rules"))
        self.assertEqual(identity["symbols"], list(SYMBOLS))
        self.assertEqual(text.frontend_runtime(), {"phonemizer": "lug-rules-1"})
        text.set_frontend("espeak")
        espeak = text.frontend_fingerprint()
        text.set_frontend("lug")
        self.assertNotEqual(text.frontend_fingerprint(), espeak)

    def test_numbers_in_words(self):
        for value, words in NUMBERS.items():
            self.assertEqual(lug_text.number(value), words, value)

    def test_expansion(self):
        for written, said in EXPANDED.items():
            self.assertEqual(lug_text.expand(written), said, written)

    def test_digits_are_read(self):
        self.assertEqual(phonemize("123 456").symbols, phonemize(lug_text.expand("123 456")).symbols)
        self.assertIn("ŋ", phonemize("Ng'ombe 2.").symbols)

    def test_expansion_is_not_in_the_fingerprint(self):
        before = text.frontend_fingerprint()
        saved = dict(lug_text.RESPELLINGS)
        try:
            lug_text.RESPELLINGS["kato"] = "kaato"
            self.assertEqual(text.frontend_fingerprint(), before)
        finally:
            lug_text.RESPELLINGS.clear()
            lug_text.RESPELLINGS.update(saved)

    def test_nothing_to_pronounce(self):
        for sentence in NOTHING:
            with self.assertRaises(NothingToPronounce, msg=sentence):
                phonemize(sentence)


@unittest.skipUnless(shutil.which("node"), "needs node to run the demo's lug.js")
class JavaScriptCopyTests(unittest.TestCase):
    """web/public/lug.js must give the ids Python gives, or the demo speaks other sounds than the model learned."""

    @classmethod
    def setUpClass(cls):
        script = ("const lug = require(process.argv[1]);"
                  "const texts = JSON.parse(process.argv[2]);"
                  "const numbers = JSON.parse(process.argv[3]);"
                  "const written = JSON.parse(process.argv[4]);"
                  "console.log(JSON.stringify({symbols: lug.SYMBOLS, said: texts.map((t) => lug.phonemize(t)),"
                  " numbers: numbers.map(lug.number), expanded: written.map(lug.expand)}));")
        done = subprocess.run(["node", "-e", script, str(LUG_JS), json.dumps(SENTENCES + NOTHING),
                               json.dumps(list(NUMBERS)), json.dumps(list(EXPANDED))],
                              capture_output=True, text=True, encoding="utf-8")
        if done.returncode:
            raise AssertionError(done.stderr)
        cls.js = json.loads(done.stdout)

    def setUp(self):
        self.previous = text.frontend()
        text.set_frontend("lug")

    def tearDown(self):
        text.set_frontend(self.previous)

    def test_same_symbols(self):
        self.assertEqual(self.js["symbols"], list(SYMBOLS))

    def test_same_ids_as_python(self):
        for sentence, said in zip(SENTENCES, self.js["said"]):
            expected = phonemize(sentence)
            self.assertEqual(said, {"ids": list(expected.ids), "stress": list(expected.stress),
                                    "text": " ".join(expected.symbols)}, sentence)

    def test_same_numbers_and_expansion_as_python(self):
        self.assertEqual(self.js["numbers"], [lug_text.number(value) for value in NUMBERS])
        self.assertEqual(self.js["expanded"], [lug_text.expand(written) for written in EXPANDED])

    def test_nothing_to_pronounce_is_null(self):
        self.assertEqual(self.js["said"][len(SENTENCES):], [None] * len(NOTHING))


if __name__ == "__main__":
    unittest.main()
