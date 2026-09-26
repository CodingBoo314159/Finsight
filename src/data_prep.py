"""
FINSIGHT - data_prep.py

Data preprocessing pipeline (Deliverable 4, Section 2). Same steps and same results as
notebooks/01_data_preprocessing.ipynb, packaged so it runs in one go without Jupyter.

Run from anywhere (paths are worked out from this file's location in the repo):
    python src/data_prep.py

Input : data/raw/bank_transactions.csv
Output: data/processed/  train.csv, val.csv, test.csv, transactions_processed.csv,
                         preprocessing_config.json

Steps:
  1. load + audit            5. resolve conflicting labels   9. synthetic SA rows (Education/Health/Utilities)
  2. drop IDs, tag debits    6. second noise sweep           10. grouped stratified 70/15/15 split
  3. map 33 -> 7 categories  7. SA localisation              11. save
  4. remove label noise      8. realistic Rand amounts
Text cleaning uses clean_description() from nlp_preprocessing.py (shared with the PDF extractor).
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

try:
    from nlp_preprocessing import clean_description
except ImportError:                      # when imported as src.data_prep
    from src.nlp_preprocessing import clean_description

# Paths relative to the repo, so this works on any computer that has the repo
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = REPO_ROOT / "data" / "raw" / "bank_transactions.csv"
OUT_DIR = REPO_ROOT / "data" / "processed"

SEED = 42

# ------------------------------------------------------------------ category mapping (33 -> 7)
CATEGORY_MAP = {
    "Supermarkets and Groceries": "Groceries",
    "Convenience Stores": "Groceries",
    "Gas Stations": "Transport",
    "Travel": "Transport",
    "Utilities": "Utilities",
    "Telecommunication Services": "Utilities",
    "Digital Entertainment": "Entertainment",
    "Arts and Entertainment": "Entertainment",
    "Restaurants": "Entertainment",            # team decision: eating out = discretionary
    "Food and Beverage Services": "Entertainment",
    "Healthcare": "Health",
    "Gyms and Fitness Centers": "Health",
    "Service": "Other Services",
    "Insurance": "Other Services",
    "Shops": "Other Services",
    "Clothing and Accessories": "Other Services",
    "Department Stores": "Other Services",
    "Bank Fees": "Other Services",
    "Bank Fee": "Other Services",
    # Education has no source rows -> generated synthetically
}
NON_SPEND = {
    "Uncategorized", "Third Party", "Transfer Credit", "Loans", "Transfer Debit",
    "Internal Account Transfer", "Payroll", "Transfer", "ATM", "Transfer Deposit",
    "Interest", "Check Deposit", "Payment", "Tax Refund",
}

# ------------------------------------------------------------------ noise filters
NOISE_PATTERN = (
    r"round up|internal account transfer|online transfer|zelle|cash app|venmo|"
    r"transfer to|transfer from|empower|float me|dividend|^check\s*\d*$|^cash\b"
)
JUNK_TEXTS = {"round", "repayment"}
FUEL_BRANDS = (r"\b(?:shell|bp|chevron|sunoco|speedway|sheetz|marathon|exxon|exxonmobil|mobil|"
               r"circle|valero|citgo|wawa|racetrac|quiktrip|qt|murphy|casey|caseys|holiday stations|"
               r"kwik trip|phillips|conoco|texaco|arco|gulf|pilot|loves|kum go)\b")
EXTRA_NOISE = (r"apple cash|bright money|instacash|afterpay|klarna|quadpay|\bzip\b|mspbna|"
               r"credit union|immediate funds|sent money|\btransfer\b|repayment")

# ------------------------------------------------------------------ SA localisation
SA_BANKS = ["absa", "fnb", "standard bank", "nedbank", "capitec", "tymebank"]
MERCHANT_MAP = {
    # Groceries
    r"walmart|wal mart|wm superc|wm supercenter|supercenter":
        ["checkers", "checkers hyper", "shoprite", "pick n pay", "spar", "boxer"],
    r"kroger|meijer|safeway|aldi|piggly wiggly|save mart|ingles markets|fine fare supermarket|instacart|grocery":
        ["pick n pay", "woolworths food", "spar", "food lovers market", "checkers", "shoprite"],
    r"target":
        ["woolworths", "woolworths food", "checkers"],
    r"star food|kwik stop|food mart|break time|enmarket|mapco|huck food|save pay":
        ["kwikspar", "spar express", "friendly grocer", "pick n pay express"],
    # Transport
    r"circle|speedway|shell oil|shell service station|shell|sunoco|chevron|bp|sheetz|holiday stations|"
    r"marathon petro|marathon|qt|quiktrip|arco|wawa|kroger fuel|kwik trip|murphy express|citgo|valero|exxon|mobil":
        ["engen", "sasol", "shell", "bp", "totalenergies", "astron energy"],
    r"uber|lyft":
        ["uber", "bolt"],
    # Entertainment
    r"dd doordash|doordash|grubhub|postmates":
        ["mr food", "uber eats"],
    r"starbucks|dunkin donuts|dunkin":
        ["vida caffe", "mugg bean", "seattle coffee", "krispy kreme"],
    r"mcdonalds|mcdonald|taco bell|wendy|chick fil|burger king|whataburger|arby|little caesars|"
    r"papa john|domino|chipotle|kfc|sonic drive|jack box|popeyes|subway":
        ["kfc", "mcdonalds", "steers", "nandos", "wimpy", "debonairs pizza", "romans pizza",
         "spur", "burger king", "chicken licken"],
    r"hulu hulu com|hulu|disney plus|netflix com netflix com|netflix":
        ["netflix", "showmax", "dstv", "disney plus"],
    r"spotify":
        ["spotify", "apple music"],
    r"\bbet\b":
        ["betway", "hollywoodbets", "supabets"],
    # Utilities
    r"gapower|duke energy|reliant energy|consumers energy|dominion energy|dte energy|comed|txu|"
    r"atmos energy|payless power|entergy|centerpoint energy|cal edison|green mountain energy|"
    r"lumbee river emc|electric cooper|dixie electric|energy":
        ["eskom", "city power", "city tshwane electricity", "city johannesburg", "ekurhuleni electricity"],
    r"water works|tucson water|sherman utility|municipal|utility":
        ["city tshwane water", "johannesburg water", "rand water", "city tshwane rates"],
    r"cal gas|spire":
        ["egoli gas"],
    r"securus inmate|att|verizon|t mobile|tmobile":
        ["vodacom", "mtn", "cell", "telkom", "rain"],
    r"rumpke":
        ["pikitup"],
    # Health
    r"kaiser|hospital|hospita|health dept|health":
        ["netcare", "mediclinic", "life healthcare", "discovery health"],
    r"dental|dent|orthod|orthodont|orthodon|dmd|smile doctors":
        ["dental studio", "smile clinic", "dental care"],
    r"planet fit|fitness|fitne|club fees":
        ["virgin active", "planet fitness", "zone fitness"],
    # Other Services
    r"dollar general|dollar tree|family dollar":
        ["mr price", "pep", "ackermans", "jet", "game"],
    r"wells fargo|chase|bank america|capital one":
        SA_BANKS,
}
US_CITIES = ["seattle", "tucson", "houston", "sacramento", "tampa", "san bernardin", "van nuys", "bronx",
             "baltimore", "el cajon", "green bay", "cincinnati", "phoenix", "san antonio", "chattanooga",
             "tallahassee", "hattiesburg", "manitowoc", "morgantown", "lafayette", "decatur", "butler",
             "york", "woodinville", "belleville", "hopewell", "eau claire", "new york", "los angeles",
             "san francisco", "chicago", "dallas", "atlanta", "miami", "denver", "las vegas", "tucker"]
SA_CITIES = ["pretoria", "centurion", "sandton", "johannesburg", "midrand", "soweto", "durban",
             "cape town", "menlyn", "rosebank", "hatfield", "bloemfontein", "polokwane", "randburg"]

_merchant_rules = [(re.compile(r"\b(?:" + p + r")\b"), opts) for p, opts in MERCHANT_MAP.items()]
_city_re = re.compile(r"\b(?:" + "|".join(sorted(US_CITIES, key=len, reverse=True)) + r")\b")

# ------------------------------------------------------------------ Rand amounts
# (median, spread, min, max) per transaction -- documented team assumptions
AMOUNT_PROFILE = {
    "Groceries":      (350, 0.9,  15, 4500),
    "Transport":      (550, 0.7,  20, 2500),
    "Entertainment":  (160, 0.8,  20, 2500),
    "Utilities":      (650, 0.8,  50, 5000),
    "Health":         (450, 0.9,  50, 8000),
    "Education":      (1800, 0.9, 100, 25000),
    "Other Services": (220, 1.0,  10, 5000),
}

# ------------------------------------------------------------------ synthetic SA rows
SYNTH_MERCHANTS = {
    "Education": ["UNISA", "UNIVERSITY OF PRETORIA", "WITS UNIVERSITY", "UNIVERSITY OF JOHANNESBURG",
                  "TSHWANE UNIVERSITY OF TECHNOLOGY", "EDUVOS", "CURRO SCHOOLS", "SPARK SCHOOLS",
                  "ADVTECH CRAWFORD", "PRETORIA BOYS HIGH", "AFRIKAANSE HOER MEISIES", "LITTLE ANGELS CRECHE",
                  "MONTESSORI PRE SCHOOL", "VAN SCHAIK BOOKSTORE", "COURSERA", "SCHOOL FEES DEPT EDUC",
                  "IIE ROSEBANK COLLEGE", "BOSTON CITY CAMPUS", "DAMELIN", "SUMMIT COLLEGE"],
    "Health":    ["CLICKS PHARMACY", "DIS-CHEM PHARMACY", "NETCARE HOSPITAL", "MEDICLINIC", "LIFE HEALTHCARE",
                  "DISCOVERY HEALTH", "MOMENTUM HEALTH", "BONITAS MEDICAL FUND", "MEDSHIELD", "GEMS MEDICAL",
                  "PATHCARE", "LANCET LABORATORIES", "AMPATH LAB", "SPEC-SAVERS", "DR NAIDOO INC",
                  "DR MOKOENA PRACTICE", "DR VAN DER MERWE", "ER24 EMERGENCY", "VIRGIN ACTIVE", "PLANET FITNESS",
                  "MEDIRITE PHARMACY", "ALPHA PHARM"],
    "Utilities": ["ESKOM", "CITY POWER JHB", "CITY OF TSHWANE", "CITY OF JOBURG", "EKURHULENI METRO",
                  "PREPAID ELECTRICITY", "PREPAID WATER", "VODACOM", "MTN", "TELKOM", "CELL C", "RAIN MOBILE",
                  "AFRIHOST", "VUMATEL", "OPENSERVE", "RAND WATER", "EGOLI GAS", "PIKITUP"],
}
SUFFIXES = {
    "Education": ["STUDENT FEES", "TUITION", "REGISTRATION", "SCHOOL FEES", "TEXTBOOKS", "EXAM FEES", "", ""],
    "Health":    ["", "", "CONTRIBUTION", "CO-PAYMENT", "SCRIPT", "CONSULTATION", "MEMBERSHIP"],
    "Utilities": ["", "", "ACCOUNT", "PREPAID", "AIRTIME", "DATA BUNDLE", "FIBRE", "RATES AND TAXES"],
}
PREFIXES = ["POS PURCHASE", "DEBIT ORDER", "EFT PAYMENT", "INTERNET PMT TO", "CARD PURCHASE",
            "MAGTAPE DEBIT", "SCHEDULED PMT", "PAYSHAP PMT", ""]
TARGET_ROWS = {"Education": 1500, "Health": 1500, "Utilities": 2000}

SPLIT_TARGETS = {"train": 0.70, "val": 0.15, "test": 0.15}


# ================================================================== steps
def load_and_audit(raw_path):
    raw = pd.read_csv(raw_path)
    audit = {
        "rows": len(raw),
        "missing_per_column": raw.isna().sum().to_dict(),
        "full_row_duplicates": int(raw.duplicated().sum()),
        "same_desc_amount_date": int(raw.duplicated(subset=["description", "amount", "txn_date"]).sum()),
        "unique_descriptions": raw["description"].nunique(),
        "raw_categories": raw["category"].nunique(),
        "date_range": (raw["txn_date"].min(), raw["txn_date"].max()),
    }
    return raw, audit


def select_spending(raw):
    """Drop IDs, tag debits, map 33 -> 7 categories, keep labelled debits, remove label noise."""
    unaccounted = set(raw["category"].dropna().unique()) - set(CATEGORY_MAP) - NON_SPEND
    if unaccounted:
        raise ValueError(f"Raw categories not mapped: {unaccounted}")
    df = raw.drop(columns=["client_id", "bank_id", "account_id", "txn_id"]).copy()
    df["txn_date"] = pd.to_datetime(df["txn_date"])
    df["direction"] = np.where(df["amount"] < 0, "debit", "credit")
    df["category_7"] = df["category"].map(CATEGORY_MAP)
    spend = df[(df["direction"] == "debit") & df["category_7"].notna()].copy()
    is_noise = spend["description"].str.contains(NOISE_PATTERN, case=False, regex=True)
    return spend[~is_noise].copy()


def clean_text(spend):
    spend["description_clean"] = spend["description"].map(clean_description)
    keep = (spend["description_clean"] != "") & ~spend["description_clean"].isin(JUNK_TEXTS)
    return spend[keep].copy()


def resolve_labels(spend):
    """Fuel stations -> Transport, then one label per clean text by majority vote (<60% dropped)."""
    is_fuel = spend["description_clean"].str.contains(FUEL_BRANDS, regex=True)
    spend.loc[is_fuel, "category_7"] = "Transport"
    grp = spend.groupby("description_clean")["category_7"]
    majority = grp.agg(lambda s: s.value_counts().index[0])
    share = grp.agg(lambda s: s.value_counts(normalize=True).iloc[0])
    spend = spend[spend["description_clean"].map(share) >= 0.6].copy()
    spend["category_7"] = spend["description_clean"].map(majority)
    is_noise2 = spend["description_clean"].str.contains(EXTRA_NOISE, regex=True)
    return spend[~is_noise2].copy()


def localise(text, rng, p_append_city=0.35):
    """Swap US merchant/city for SA equivalents. Returns (clean_sa_text, merchant_was_swapped)."""
    t = _city_re.sub(lambda m: str(rng.choice(SA_CITIES)), text)
    swapped = False
    for rx, options in _merchant_rules:
        if rx.search(t):
            t = rx.sub("\x00", t)
            t = t.replace("\x00", " " + str(rng.choice(options)) + " ", 1).replace("\x00", " ")
            swapped = True
            break
    if not any(c in t for c in SA_CITIES) and rng.random() < p_append_city:
        t = f"{t} {rng.choice(SA_CITIES)}"
    return clean_description(t), swapped


def localise_all(spend, seed=SEED):
    loc_rng = np.random.default_rng(seed)
    results = [localise(t, loc_rng) for t in spend["description_clean"]]
    spend["description_sa"] = [r[0] for r in results]
    spend["merchant_localised"] = [r[1] for r in results]
    return spend


def to_rand(abs_amounts, category):
    """Map each amount's rank within its category onto a realistic Rand log-normal curve."""
    med, sigma, lo, hi = AMOUNT_PROFILE[category]
    p = abs_amounts.rank(pct=True, method="average").clip(0.005, 0.995)
    zar = np.exp(np.log(med) + sigma * norm.ppf(p))
    return zar.clip(lo, hi).round(2)


def add_rand_amounts(spend):
    spend["amount_zar"] = 0.0
    for cat, idx in spend.groupby("category_7").groups.items():
        spend.loc[idx, "amount_zar"] = -to_rand(spend.loc[idx, "amount"].abs(), cat)
    return spend


def make_synthetic_rows(spend, seed=SEED + 1):
    aug_rng = np.random.default_rng(seed)
    sa_cities_up = [c.upper() for c in SA_CITIES]

    def synth_description(cat):
        merchant = str(aug_rng.choice(SYNTH_MERCHANTS[cat]))
        parts = [str(aug_rng.choice(PREFIXES)), merchant, str(aug_rng.choice(SUFFIXES[cat]))]
        if aug_rng.random() < 0.5:
            parts.append(str(aug_rng.choice(sa_cities_up)))
        if aug_rng.random() < 0.5:
            parts.append(f"{aug_rng.integers(1000, 9999)}*{aug_rng.integers(1000, 9999)}")
        if aug_rng.random() < 0.4:
            parts.append(f"{aug_rng.integers(1, 29):02d} {aug_rng.choice(['JUN','JUL','AUG','SEP'])}")
        return " ".join(p for p in parts if p), merchant

    rows = []
    for cat, target in TARGET_ROWS.items():
        n_new = max(0, target - int((spend["category_7"] == cat).sum()))
        for _ in range(n_new):
            desc, merchant = synth_description(cat)
            rows.append({"description": desc, "category_7": cat, "merchant_group": "syn_" + merchant})

    synth = pd.DataFrame(rows)
    synth["description_clean"] = synth["description"].map(clean_description)
    synth["description_sa"] = synth["description_clean"]
    synth["merchant_localised"] = True
    synth["is_synthetic"] = True
    synth["txn_date"] = pd.to_datetime("2023-06-01") + pd.to_timedelta(aug_rng.integers(0, 122, len(synth)), unit="D")
    synth["amount_zar"] = 0.0
    for cat, idx in synth.groupby("category_7").groups.items():
        synth.loc[idx, "amount_zar"] = -to_rand(pd.Series(aug_rng.random(len(idx)), index=idx), cat)
    return synth


def combine(spend, synth):
    spend["is_synthetic"] = False
    spend["merchant_group"] = spend["description_clean"]
    cols = ["txn_date", "description", "description_clean", "description_sa", "amount_zar",
            "category_7", "is_synthetic", "merchant_localised", "merchant_group"]
    return pd.concat([spend[cols], synth[cols]], ignore_index=True)


def split_data(data, seed=SEED):
    """Greedy stratified group split: every merchant group stays in one split, 70/15/15 per category."""
    if data.groupby("merchant_group")["category_7"].nunique().max() != 1:
        raise ValueError("A merchant group has more than one category")
    group_sizes = data.groupby(["category_7", "merchant_group"]).size()
    split_rng = np.random.default_rng(seed)
    split_of = {}
    for cat, sizes in group_sizes.groupby(level=0):
        sizes = sizes.droplevel(0)
        sizes = sizes.sample(frac=1, random_state=split_rng.integers(1_000_000))
        sizes = sizes.sort_values(ascending=False, kind="stable")
        goal = {s: t * sizes.sum() for s, t in SPLIT_TARGETS.items()}
        filled = {s: 0 for s in SPLIT_TARGETS}
        for grp, n in sizes.items():
            s = max(SPLIT_TARGETS, key=lambda k: (goal[k] - filled[k]) / goal[k])
            split_of[grp] = s
            filled[s] += n
    data["split"] = data["merchant_group"].map(split_of)
    train_texts = set(data.loc[data["split"] == "train", "description_sa"])
    data["seen_in_train"] = data["description_sa"].isin(train_texts) & (data["split"] != "train")
    return data


def save_outputs(data, audit, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = ["txn_date", "description", "description_sa", "amount_zar", "category_7",
            "is_synthetic", "merchant_localised", "merchant_group", "split", "seen_in_train"]
    final = data[keep].rename(columns={"description": "description_raw", "category_7": "category"})
    final.to_csv(out_dir / "transactions_processed.csv", index=False)
    for s in ["train", "val", "test"]:
        final[final["split"] == s].to_csv(out_dir / f"{s}.csv", index=False)
    config = {
        "seed": SEED,
        "category_map": CATEGORY_MAP,
        "non_spend_categories": sorted(NON_SPEND),
        "noise_patterns": [NOISE_PATTERN, EXTRA_NOISE],
        "merchant_map": MERCHANT_MAP,
        "amount_profile_zar": AMOUNT_PROFILE,
        "synthetic_targets": TARGET_ROWS,
        "raw_audit": audit,
        "final_counts": final.groupby(["category", "split"]).size().unstack().to_dict(),
    }
    with open(out_dir / "preprocessing_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)
    return final


def run_pipeline(raw_path=RAW_PATH, out_dir=OUT_DIR, verbose=True):
    """Run every step and save the outputs. Returns the final DataFrame."""
    raw_path, out_dir = Path(raw_path), Path(out_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_path}. Is data/raw/bank_transactions.csv in the repo?")
    log = print if verbose else (lambda *a, **k: None)

    raw, audit = load_and_audit(raw_path)
    log(f"Loaded {len(raw):,} raw rows")
    spend = select_spending(raw)
    log(f"Labelled spending rows after noise filter: {len(spend):,}")
    spend = clean_text(spend)
    log(f"After text cleaning: {len(spend):,}")
    spend = resolve_labels(spend)
    log(f"After label resolution + 2nd noise sweep: {len(spend):,}")
    spend = localise_all(spend)
    spend = add_rand_amounts(spend)
    synth = make_synthetic_rows(spend)
    log(f"Synthetic SA rows added: {len(synth):,}")
    data = split_data(combine(spend, synth))
    final = save_outputs(data, audit, out_dir)

    log(f"Final dataset: {len(final):,} rows -> {out_dir}")
    log((final["split"].value_counts(normalize=True) * 100).round(1).to_string())
    return final


if __name__ == "__main__":
    run_pipeline()
