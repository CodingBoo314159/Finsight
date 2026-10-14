"""
FINSIGHT - pdf_generator.py  (v2)

Makes synthetic South African bank statement PDFs to test the extractor, plus a ground-truth CSV
per statement recording exactly what was printed.

v2 changes (to match notebooks/01_data_preprocessing.ipynb v2):
- the 9 v2 categories (Groceries, Money Sent, Banking, Food, Transport, Shopping, Entertainment, Other, Utilities)
- the printed description IS description_sa (full statement line, no cleaning, no added prefixes)
- amounts are amount_zar, dates are the real processing dates of the test rows (Jun-Sep 2023),
  so purchase dates inside the descriptions stay consistent
- no dependency on nlp_preprocessing.py any more

Three FICTIONAL banks with different layouts (grid lines / no lines / striped rows), date formats and
amount conventions. Spending rows come from the TEST split only, without replacement, so the classifier
has never seen them. Every page says "SYNTHETIC TEST DOCUMENT".

Usage:
    from pdf_generator import generate_statements
    index, truth = generate_statements("data/processed/test.csv", "data/finsight_statements")
"""

import calendar
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

SEED = 42
N_STATEMENTS = 30          # 10 per bank

BANKS = {
    "protea": {   # grid lines, "02 Sep" dates, one signed Amount column with Cr suffix
        "name": "Protea Bank", "colour": "#1F5F4A", "style": "ruled",
        "date_fmt": "%d %b", "columns": ["Date", "Description", "Amount", "Balance"],
        "col_mm": [20, 100, 30, 30],
    },
    "karoo": {    # no lines, dd/mm/yyyy, Money In / Money Out, space as thousands separator
        "name": "Karoo Savings Bank", "colour": "#8A3B12", "style": "open",
        "date_fmt": "%d/%m/%Y", "columns": ["Date", "Description", "Money In", "Money Out", "Balance"],
        "col_mm": [24, 84, 24, 24, 24],
    },
    "highveld": { # striped rows, ISO dates, Debits / Credits
        "name": "Highveld Mutual", "colour": "#1F3864", "style": "zebra",
        "date_fmt": "%Y-%m-%d", "columns": ["Date", "Description", "Debits", "Credits", "Balance"],
        "col_mm": [24, 84, 24, 24, 24],
    },
}

# average spending rows per category per month (roughly the mix in the test split)
MONTHLY_COUNTS = {"Groceries": 9, "Money Sent": 7, "Banking": 6, "Food": 6, "Transport": 3,
                  "Shopping": 2, "Entertainment": 2, "Other": 1, "Utilities": 1}

CUSTOMERS = ["MR T MOKOENA", "MS L VAN WYK", "MR K NAIDOO", "MS S DLAMINI", "MR A PIETERSEN",
             "MS N MAHLANGU", "MR J BOTHA", "MS P GOVENDER", "MR S NKOSI", "MS R SMITH"]
EMPLOYERS = ["ACACIA LOGISTICS", "MOTSWEDI HOLDINGS", "BLUE CRANE SYSTEMS", "IMPALA RETAIL GROUP",
             "SUNVELD ENGINEERING", "BAOBAB CONSULTING"]


# ---------------------------------------------------------------- transactions
def build_statement(stmt_id, bank_key, year, month, pool, rng):
    """pool = {(month, category): list of unused test rows}. Returns (rows DataFrame, meta dict)."""
    n_days = calendar.monthrange(year, month)[1]
    rows = []

    def add(day_date, text, amount, category, is_spend, source):
        rows.append({"date": day_date, "description_sa": " ".join(str(text).split()),
                     "amount": round(float(amount), 2), "category": category,
                     "is_spend": is_spend, "source": source})

    # salary on the 25th (not spending, the classifier skips it)
    add(date(year, month, 25), f"SALARY {rng.choice(EMPLOYERS)}", round(rng.uniform(15_000, 45_000), -2),
        "Income", False, "generated")

    # spending rows: real test-split rows from this month, printed exactly as description_sa
    for cat, lam in MONTHLY_COUNTS.items():
        key = (month, cat)
        for _ in range(max(1, rng.poisson(lam))):
            if not pool.get(key):
                break
            r = pool[key].pop()
            add(pd.Timestamp(r["txn_date"]).date(), r["description_sa"], -abs(r["amount_zar"]), cat, True, "test_split")

    # sometimes a transfer to savings (not spending)
    if rng.random() < 0.6:
        add(date(year, month, 26), "TRANSFER TO SAVINGS", -round(rng.uniform(500, 3_000), -2),
            "Transfer", False, "generated")

    df = pd.DataFrame(rows)
    df["order"] = rng.permutation(len(df))
    df = df.sort_values(["date", "order"]).drop(columns="order").reset_index(drop=True)

    # opening balance: enough to cover spending until payday plus a buffer, ~15% dip into overdraft
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
    cols = ["statement_id", "row_no", "date", "description_sa", "amount", "balance", "category", "is_spend", "source"]
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
    desc = lambda t: Paragraph(escape(t), DESC_STYLE)        # escape & < > so reportlab prints them as text
    if meta["bank"] == "protea":
        rows.append([first_day, "BALANCE BROUGHT FORWARD", "", fmt_money(opening, "protea", "balance")])
        for r in df.itertuples():
            rows.append([r.date.strftime(b["date_fmt"]), desc(r.description_sa),
                         fmt_money(r.amount, "protea", "amount"), fmt_money(r.balance, "protea", "balance")])
    else:
        rows.append([first_day, "OPENING BALANCE", "", "", fmt_money(opening, meta["bank"], "balance")])
        for r in df.itertuples():
            money_in = fmt_money(r.amount, meta["bank"], "in") if r.amount > 0 else ""
            money_out = fmt_money(r.amount if meta["bank"] == "karoo" else -r.amount, meta["bank"], "out") if r.amount < 0 else ""
            cells = [money_in, money_out] if meta["bank"] == "karoo" else [money_out, money_in]  # highveld: Debits, Credits
            rows.append([r.date.strftime(b["date_fmt"]), desc(r.description_sa), *cells,
                         fmt_money(r.balance, meta["bank"], "balance")])
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
    else:
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
    """Make n_statements PDFs + ground truth in out_dir. Returns (index, truth) DataFrames."""
    out_dir = Path(out_dir)
    (out_dir / "pdfs").mkdir(parents=True, exist_ok=True)
    (out_dir / "ground_truth").mkdir(parents=True, exist_ok=True)
    for old in list((out_dir / "pdfs").glob("*.pdf")) + list((out_dir / "ground_truth").glob("*.csv")):
        old.unlink()                                          # clear the old v1 statements first

    test = pd.read_csv(test_csv, parse_dates=["txn_date"])
    rng = np.random.default_rng(seed)
    shuffled = test.sample(frac=1, random_state=seed)
    pool = {(m, c): g.to_dict("records") for (m, c), g in shuffled.groupby([shuffled.txn_date.dt.month, "category"])}
    months = sorted(test.txn_date.dt.to_period("M").unique())

    bank_keys = list(BANKS)
    index_rows, all_truth = [], []
    for i in range(n_statements):
        stmt_id = f"STMT{i+1:03d}"
        bank_key = bank_keys[i % len(bank_keys)]
        p = months[i % len(months)]
        df, meta = build_statement(stmt_id, bank_key, p.year, p.month, pool, rng)

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
    idx, gt = generate_statements("data/processed/test.csv", "data/finsight_statements")
    print(f"Generated {len(idx)} statements, {len(gt):,} transactions -> data/finsight_statements")
