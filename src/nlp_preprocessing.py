"""
FINSIGHT - nlp_preprocessing.py

Shared text cleaning for transaction descriptions.

The SAME function must be used on:
  1. the training data (notebooks/01_data_preprocessing.ipynb)
  2. the synthetic SA rows generated for Education / Health / Utilities
  3. text extracted from PDF bank statements (src/pdf_extraction.py, app.py)
so that the classifier sees identically formatted text at training and at prediction time.

Note: TF-IDF vectorisation is NOT done here. The vectoriser is fitted inside the
classifier pipeline (src/ml_classifier.py) on the training split only, to avoid leakage.
"""

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

__all__ = ["clean_description", "BANK_BOILERPLATE", "STOP_WORDS"]

# Anonymisation token used in the public source dataset (sometimes pasted inside words)
PLACEHOLDER_NAMES = ["maryse", "hemant"]

# Banking filler words that say HOW a payment was made, not WHAT it was for.
# Longest phrases are matched first (see _boilerplate_re).
BANK_BOILERPLATE = [
    # US source-data wording
    "purchase authorized on", "authorized on", "point of sale", "debit card purchase",
    "visa check card", "recurring payment", "recurring debit", "card purchase",
    "debit purchase", "pos debit", "pos purchase", "check card", "checkcard",
    "withdrawal", "purchase", "recurring", "debit", "card", "visa", "pos",
    "pymt", "pmt", "payment", "web id", "co id", "indn", "des", "ppd", "ccd",
    "ach", "usa", "transaction", "sale", "mobile", "online",
    # South African statement wording (synthetic rows and PDF statements)
    "debit order", "internet pmt to", "eft payment", "scheduled pmt", "payshap pmt",
    "magtape", "eft", "payshap", "internet", "scheduled", "order",
]

US_STATES = set("""al ak az ar ca co ct de fl ga hi id il in ia ks ky la me md ma mi mn ms mo
mt ne nv nh nj nm ny nc nd oh ok or pa ri sc sd tn tx ut vt va wa wv wi wy dc us""".split())

MONTHS = set("jan feb mar apr may jun jul aug sep sept oct nov dec".split())

STOP_WORDS = set(ENGLISH_STOP_WORDS) | US_STATES | MONTHS

_boilerplate_re = re.compile(
    r"\b(?:" + "|".join(sorted(map(re.escape, BANK_BOILERPLATE), key=len, reverse=True)) + r")\b"
)
_names_re = re.compile("|".join(PLACEHOLDER_NAMES))


def clean_description(text):
    """Reduce a raw transaction description to its merchant words.

    Steps: lowercase -> remove anonymisation token -> replace every non-letter
    with a space (digits, dates, card numbers, auth codes, punctuation) ->
    remove banking boilerplate -> drop single letters, stop words, US state
    codes and month names.

    >>> clean_description("DEBIT ORDER UNISA STUDENT FEES 4521*8876 02 SEP")
    'unisa student fees'
    """
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = _names_re.sub(" ", t)
    t = re.sub(r"[^a-z]+", " ", t)
    t = _boilerplate_re.sub(" ", t)
    tokens = [w for w in t.split() if len(w) > 1 and w not in STOP_WORDS]
    return " ".join(tokens)


# Test cases shared with notebook 01 (Cell 8). Run: python src/nlp_preprocessing.py
TEST_CASES = {
    "PURCHASE AUTHORIZED ON 09/01 PLAYSTATION NETWOR 1036 CA S583244812385003 111": "playstation networ",
    "Pos Debit-    2975 2975 Jack In The Box 16 Tucson AZ": "jack box tucson",
    "MCDONALD'S F11823 BRIDGEPORT CT 07/07": "mcdonald bridgeport",
    "DEBIT ORDER UNISA STUDENT FEES 4521*8876 02 SEP": "unisa student fees",
    "EFT PAYMENT DIS-CHEM PHARMACY MENLYN 24 AUG": "dis chem pharmacy menlyn",
    "EFT S/C": "",
    "ORDER": "",
    None: "",
}

if __name__ == "__main__":
    failures = 0
    for raw, expected in TEST_CASES.items():
        got = clean_description(raw)
        ok = got == expected
        failures += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {raw!r:80} -> {got!r}")
    print("\nAll tests passed." if failures == 0 else f"\n{failures} test(s) FAILED.")
