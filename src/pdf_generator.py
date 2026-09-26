"""
FINSIGHT - pdf_generator.py

Generates synthetic South African bank statement PDFs for testing the extraction module,
plus a ground-truth CSV per statement recording exactly what was printed.

- Three FICTIONAL banks with different layouts (ruled grid / open / zebra rows), different date
  formats and different amount conventions. No real bank branding is imitated.
- Spending rows are drawn without replacement from the TEST split, so the PDFs contain
  transactions the classifier never saw during training.
- Every page is marked "SYNTHETIC TEST DOCUMENT".

Usage:
    python src/pdf_generator.py                       # repo paths, 30 statements
    from src.pdf_generator import generate_statements
    index, truth = generate_statements("data/processed/test.csv", "data/synthetic_statements")
"""

import calendar
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

try:
    from nlp_preprocessing import clean_description
except ImportError:                      # when imported as src.pdf_generator
    from src.nlp_preprocessing import clean_description

SEED = 42

# ---------------------------------------------------------------- settings
# Fictional bank names on purpose: we must not imitate real banks' branding.
# Each bank has a different layout so the extractor is tested on several formats.
BANKS = {
    "protea": {   # ruled table (grid lines), "02 Sep" dates, one signed Amount column with Cr suffix
        "name": "Protea Bank", "colour": "#1F5F4A", "style": "ruled",
        "date_fmt": "%d %b", "columns": ["Date", "Description", "Amount", "Balance"],
        "col_mm": [20, 100, 30, 30],
    },
    "karoo": {    # open table (no grid), dd/mm/yyyy, Money In / Money Out, space thousands
        "name": "Karoo Savings Bank", "colour": "#8A3B12", "style": "open",
        "date_fmt": "%d/%m/%Y", "columns": ["Date", "Description", "Money In", "Money Out", "Balance"],
        "col_mm": [24, 84, 24, 24, 24],
    },
    "highveld": { # zebra rows, ISO dates, Debits / Credits
        "name": "Highveld Mutual", "colour": "#1F3864", "style": "zebra",
        "date_fmt": "%Y-%m-%d", "columns": ["Date", "Description", "Debits", "Credits", "Balance"],
        "col_mm": [24, 84, 24, 24, 24],
    },
}

# Expected number of spending transactions per category per month (team assumption)
MONTHLY_COUNTS = {"Groceries": 12, "Transport": 8, "Entertainment": 10, "Other Services": 6,
                  "Utilities": 3, "Health": 2, "Education": 1}
DEBIT_ORDER_CATS = {"Utilities", "Health", "Education"}   # usually paid by debit order early in the month
POS_PREFIXES = ["POS PURCHASE", "CARD PURCHASE", "POS PURCHASE", ""]
DO_PREFIXES = ["DEBIT ORDER", "DEBIT ORDER", "MAGTAPE DEBIT", "INTERNET PMT TO", "EFT PAYMENT"]

CUSTOMERS = ["MR T MOKOENA", "MS L VAN WYK", "MR K NAIDOO", "MS S DLAMINI", "MR A PIETERSEN",
             "MS N MAHLANGU", "MR J BOTHA", "MS P GOVENDER", "MR S NKOSI", "MS R SMITH"]
EMPLOYERS = ["ACACIA LOGISTICS", "MOTSWEDI HOLDINGS", "BLUE CRANE SYSTEMS", "IMPALA RETAIL GROUP",
             "SUNVELD ENGINEERING", "BAOBAB CONSULTING"]
STATEMENT_MONTHS = [(2026, m) for m in range(4, 10)]   # Apr-Sep 2026
N_STATEMENTS = 30                                      # 10 per bank


# ---------------------------------------------------------------- transactions
def build_statement(stmt_id, bank_key, year, month, pool, rng):
    """Return (rows DataFrame, meta dict). pool = dict of category -> list of unused test rows."""
    n_days = calendar.monthrange(year, month)[1]
    rows = []

    def add(day, text, amount, category, is_spend, source, expected=None):
        text = " ".join(text.split())
        rows.append({"day": int(day), "description_pdf": text,
                     "description_sa": expected if expected is not None else clean_description(text),
                     "amount": round(float(amount), 2), "category": category,
                     "is_spend": is_spend, "source": source})

    # Income: salary on the 25th
    add(25, f"SALARY {rng.choice(EMPLOYERS)}", round(rng.uniform(15_000, 45_000), -2), "Income", False, "generated")

    # Spending: drawn WITHOUT replacement from the test split (unseen by the classifier)
    for cat, lam in MONTHLY_COUNTS.items():
        for _ in range(max(1 if cat != "Education" else 0, rng.poisson(lam))):
            if not pool[cat]:
                break
            r = pool[cat].pop()
            if cat in DEBIT_ORDER_CATS and rng.random() < 0.6:
                day, prefix = rng.integers(1, 8), rng.choice(DO_PREFIXES)
            else:
                day, prefix = rng.integers(1, n_days + 1), rng.choice(POS_PREFIXES)
            noise = ""
            if rng.random() < 0.5:
                noise += f" {rng.integers(1000, 9999)}*{rng.integers(1000, 9999)}"
            if rng.random() < 0.3:
                noise += f" {date(year, month, int(day)).strftime('%d %b').upper()}"
            add(day, f"{prefix} {r['description_sa'].upper()}{noise}", -abs(r["amount_zar"]), cat, True,
                "test_split", expected=r["description_sa"])

    # Bank fee and (sometimes) a transfer to savings
    add(n_days, "MONTHLY ACCOUNT FEE", -float(rng.choice([65, 69, 75, 99])), "Other Services", True, "generated")
    if rng.random() < 0.6:
        add(26, "TRANSFER TO SAVINGS", -round(rng.uniform(500, 3_000), -2), "Transfer", False, "generated")

    df = pd.DataFrame(rows)
    df["order"] = rng.permutation(len(df))                   # random order within the same day
    df = df.sort_values(["day", "order"]).drop(columns="order").reset_index(drop=True)
    df["date"] = [date(year, month, d) for d in df["day"]]

    # Opening balance = money left from last month's salary: enough to cover spending until
    # payday plus a buffer; ~15% of customers dip into overdraft (tests "Dr"/negative balances)
    lowest_point = df["amount"].cumsum().min()
    buffer = rng.uniform(200, 6_000) - (2_500 if rng.random() < 0.15 else 0)
    opening = round(max(0.0, -lowest_point) + buffer, 2)
    df["balance"] = (opening + df["amount"].cumsum()).round(2)
    df.insert(0, "statement_id", stmt_id)
    df.insert(1, "row_no", range(1, len(df) + 1))

    meta = {"statement_id": stmt_id, "bank": bank_key, "bank_name": BANKS[bank_key]["name"],
            "layout": BANKS[bank_key]["style"], "customer": str(rng.choice(CUSTOMERS)),
            "account_no": f"{rng.integers(10**9, 10**10)}",
            "period_start": date(year, month, 1).isoformat(), "period_end": date(year, month, n_days).isoformat(),
            "opening_balance": opening, "closing_balance": float(df["balance"].iloc[-1]),
            "n_transactions": len(df)}
    cols = ["statement_id", "row_no", "date", "description_pdf", "description_sa", "amount", "balance",
            "category", "is_spend", "source"]
    return df[cols], meta


# ---------------------------------------------------------------- rendering
def fmt_money(x, bank_key, kind):
    """kind: 'amount' (signed single column), 'in', 'out', 'balance'."""
    if bank_key == "protea":                                  # 1,234.56 / 1,234.56Cr
        if kind == "amount":
            return f"{abs(x):,.2f}" + ("Cr" if x > 0 else "")
        return f"{abs(x):,.2f}" + ("Cr" if x >= 0 else "Dr")
    if bank_key == "karoo":                                   # 1 234.56 / -1 234.56
        return f"{x:,.2f}".replace(",", " ")
    return f"{x:,.2f}"                                        # highveld: 1,234.56 / -1,234.56

DESC_STYLE = ParagraphStyle("desc", fontName="Helvetica", fontSize=8, leading=10)

def table_rows(df, meta):
    b = BANKS[meta["bank"]]
    first_day = date.fromisoformat(meta["period_start"]).strftime(b["date_fmt"])
    rows = [b["columns"]]
    opening = meta["opening_balance"]
    if meta["bank"] == "protea":
        rows.append([first_day, "BALANCE BROUGHT FORWARD", "", fmt_money(opening, "protea", "balance")])
        for r in df.itertuples():
            rows.append([r.date.strftime(b["date_fmt"]), Paragraph(r.description_pdf, DESC_STYLE),
                         fmt_money(r.amount, "protea", "amount"), fmt_money(r.balance, "protea", "balance")])
    else:
        rows.append([first_day, "OPENING BALANCE", "", "", fmt_money(opening, meta["bank"], "balance")])
        for r in df.itertuples():
            money_in = fmt_money(r.amount, meta["bank"], "in") if r.amount > 0 else ""
            money_out = fmt_money(r.amount if meta["bank"] == "karoo" else -r.amount, meta["bank"], "out") if r.amount < 0 else ""
            cells = [money_in, money_out] if meta["bank"] == "karoo" else [money_out, money_in]  # highveld: Debits, Credits
            rows.append([r.date.strftime(b["date_fmt"]), Paragraph(r.description_pdf, DESC_STYLE),
                         *cells, fmt_money(r.balance, meta["bank"], "balance")])
    return rows

def render_pdf(df, meta, path):
    b = BANKS[meta["bank"]]
    colour = colors.HexColor(b["colour"])
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=15*mm, rightMargin=15*mm,
                            topMargin=15*mm, bottomMargin=18*mm,
                            title=f"{b['name']} statement {meta['statement_id']}", author=b["name"])
    h = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=16, textColor=colour, leading=20)
    s = ParagraphStyle("s", fontName="Helvetica", fontSize=9, leading=12)
    story = [
        Paragraph(b["name"], h),
        Paragraph("Statement of Account", s),
        Spacer(1, 4*mm),
        Paragraph(f"<b>Account holder:</b> {meta['customer']}", s),
        Paragraph(f"<b>Account number:</b> ******{meta['account_no'][-4:]}", s),
        Paragraph(f"<b>Statement period:</b> {meta['period_start']} to {meta['period_end']}", s),
        Paragraph(f"<b>Opening balance:</b> R {meta['opening_balance']:,.2f} &nbsp;&nbsp; "
                  f"<b>Closing balance:</b> R {meta['closing_balance']:,.2f}", s),
        Spacer(1, 5*mm),
    ]
    t = Table(table_rows(df, meta), colWidths=[w*mm for w in b["col_mm"]], repeatRows=1)
    style = [
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
        ("FONT", (0, 1), (-1, -1), "Helvetica", 8),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), colour),
        ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
    if b["style"] == "ruled":
        style += [("GRID", (0, 0), (-1, -1), 0.4, colors.grey)]
    elif b["style"] == "open":
        style += [("LINEBELOW", (0, 0), (-1, 0), 0.8, colour)]
    else:  # zebra
        style += [("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF2F8")])]
    t.setStyle(TableStyle(style))
    story.append(t)

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        canvas.drawString(15*mm, 10*mm, "SYNTHETIC TEST DOCUMENT - FINSIGHT (ITDPA3-34). NOT A REAL BANK STATEMENT.")
        canvas.drawRightString(A4[0] - 15*mm, 10*mm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def generate_statements(test_csv, out_dir, n_statements=N_STATEMENTS, seed=SEED):
    """Generate n_statements PDFs + ground truth into out_dir. Returns (index, truth) DataFrames."""
    out_dir = Path(out_dir)
    (out_dir / "pdfs").mkdir(parents=True, exist_ok=True)
    (out_dir / "ground_truth").mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(test_csv)

    gen_rng = np.random.default_rng(seed)
    shuffled = test.sample(frac=1, random_state=seed)
    pool = {cat: g.to_dict("records") for cat, g in shuffled.groupby("category")}

    bank_keys = list(BANKS)
    index_rows, all_truth = [], []
    for i in range(n_statements):
        stmt_id = f"STMT{i+1:03d}"
        bank_key = bank_keys[i % len(bank_keys)]
        year, month = STATEMENT_MONTHS[gen_rng.integers(len(STATEMENT_MONTHS))]
        df, meta = build_statement(stmt_id, bank_key, year, month, pool, gen_rng)

        pdf_path = out_dir / "pdfs" / f"{stmt_id}_{bank_key}.pdf"
        gt_path = out_dir / "ground_truth" / f"{stmt_id}_{bank_key}.csv"
        render_pdf(df, meta, pdf_path)
        df.to_csv(gt_path, index=False)

        meta.update(pdf_file=pdf_path.name, ground_truth_file=gt_path.name)
        index_rows.append(meta)
        all_truth.append(df)

    index = pd.DataFrame(index_rows)
    truth = pd.concat(all_truth, ignore_index=True)
    index.to_csv(out_dir / "statements_index.csv", index=False)
    truth.to_csv(out_dir / "ground_truth_all.csv", index=False)
    return index, truth


if __name__ == "__main__":
    idx, gt = generate_statements("data/processed/test.csv", "data/synthetic_statements")
    print(f"Generated {len(idx)} statements, {len(gt):,} transactions -> data/synthetic_statements")
