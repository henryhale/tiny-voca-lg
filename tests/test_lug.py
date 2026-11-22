"""The Luganda front end ("lug"), and its JavaScript copy for the browser demo.

The released tiny-voca-lg checkpoint was trained on exactly these tokens: the
expected outputs below were recorded before its training started.
"""

import json
from pathlib import Path
import shutil
import subprocess
import unittest

from tiny_voca import lug, text
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
]
NOTHING = ["こんにちは、世界。", "!!! ...", "\U0001F642", "123 456"]


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
        self.assertEqual(lug.word_phones("2026"), [])                                 # digits are not read

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
                  "console.log(JSON.stringify({symbols: lug.SYMBOLS, said: texts.map((t) => lug.phonemize(t))}));")
        done = subprocess.run(["node", "-e", script, str(LUG_JS), json.dumps(SENTENCES + NOTHING)],
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

    def test_nothing_to_pronounce_is_null(self):
        self.assertEqual(self.js["said"][len(SENTENCES):], [None] * len(NOTHING))


if __name__ == "__main__":
    unittest.main()
