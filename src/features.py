"""
FINSIGHT - features.py  (v2)

Turns extracted statement rows into the same columns the classifier was trained on
in notebooks/01_data_preprocessing.ipynb (cell 8 and cell 9).

Input  : a DataFrame with  date (statement / processing date), description_sa, amount
Output : the same rows plus
         stmt_year, stmt_month, stmt_day,
         purchase_year, purchase_month, purchase_day, days_to_process, has_purchase_date,
         hour, minute, has_time, is_online, txn_type, amount_zar

The rules are copied from notebook 01 so training data and PDF data get identical features.
Do NOT add text cleaning here: the model is trained on the full description_sa text.
"""

import re

import pandas as pd

__all__ = ["add_features", "txn_type", "purchase_date"]

DATE_IN_TEXT = re.compile(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b")
TIME_IN_TEXT = r"\b(\d{1,2}):(\d{2})\s*(AM|PM)?\b"
ONLINE = r"www|https?:|\.com|\bcom\b|\.net\b|\.co\b"


def purchase_date(text, stmt_date):
    """Date written inside the description (month/day), if it is 0-10 days before the statement date."""
    m = DATE_IN_TEXT.search(text)
    if not m:
        return pd.NaT
    month, day = int(m.group(1)), int(m.group(2))
    try:
        d = pd.Timestamp(year=stmt_date.year, month=month, day=day)
    except ValueError:
        return pd.NaT                       # not a real date
    if d > stmt_date:
        d = d - pd.DateOffset(years=1)      # bought in december, processed in january
    gap = (stmt_date - d).days
    return d if 0 <= gap <= 10 else pd.NaT


def txn_type(text):
    t = text.lower()
    if "atm" in t or "withdrawal" in t:
        return "atm"
    if "recurring" in t:
        return "recurring"
    if "point of sale" in t or "pos " in t or "purchase" in t or "debit card" in t or "checkcard" in t or "check card" in t:
        return "card"
    if "ach" in t or "ppd" in t or "web id" in t or "bill pay" in t or "autopay" in t or "payment" in t:
        return "bill_payment"
    if "transfer" in t or "xfer" in t:
        return "transfer"
    return "other"


def add_features(df, date_col="date", text_col="description_sa", amount_col="amount"):
    out = df.copy()
    when = pd.to_datetime(out[date_col])
    text = out[text_col].fillna("").astype(str)

    out["stmt_year"], out["stmt_month"], out["stmt_day"] = when.dt.year, when.dt.month, when.dt.day

    pdate = pd.Series([purchase_date(t, w) for t, w in zip(text, when)], index=out.index, dtype="datetime64[ns]")
    out["purchase_year"] = pdate.dt.year.astype("Int64")
    out["purchase_month"] = pdate.dt.month.astype("Int64")
    out["purchase_day"] = pdate.dt.day.astype("Int64")
    out["days_to_process"] = (when - pdate).dt.days.astype("Int64")
    out["has_purchase_date"] = pdate.notna().astype(int)

    t = text.str.extract(TIME_IN_TEXT, flags=re.I)
    hour = pd.to_numeric(t[0], errors="coerce")
    hour = hour.where(~((t[2].str.upper() == "PM") & (hour < 12)), hour + 12)
    hour = hour.where(~((t[2].str.upper() == "AM") & (hour == 12)), 0)
    out["hour"] = hour.astype("Int64")
    out["minute"] = pd.to_numeric(t[1], errors="coerce").astype("Int64")
    out["has_time"] = out["hour"].notna().astype(int)

    out["is_online"] = text.str.contains(ONLINE, case=False, regex=True).astype(int)
    out["txn_type"] = text.map(txn_type)
    out["amount_zar"] = out[amount_col].abs().round(2)
    return out
