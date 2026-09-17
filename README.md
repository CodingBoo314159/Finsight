# FINSIGHT

An AI-driven personal financial wellness analytics system for South African households. FINSIGHT extracts transactions from bank statement PDFs, classifies them into spending categories using a hybrid rule-based + machine learning approach, and presents spending insights through an interactive dashboard.

Built for ITDPA3-34 at Eduvos, Pretoria.

## Team

| Name | Focus area |
|---|---|
| Jennifer Ann Kok | ML classifier, Gradio interface, rule-based categorisation |
| Kimberly Marufu | *(add focus area)* |
| Thabang Manyama | *(add focus area)* |
| Langalami Langa | *(add focus area)* |

## Project overview

FINSIGHT takes a synthetic PDF bank statement, extracts the transaction data, and runs it through a hybrid classification pipeline: a rule-based layer handles transactions with clear merchant matches, and anything left over is passed to a trained ML classifier. Classified transactions are aggregated into rule-based financial summaries and displayed as visualisations on a Gradio dashboard, deployed privately on Hugging Face Spaces.

To protect privacy, the system uses only publicly available and synthetically generated data — no real bank statements or banking credentials are involved at any stage.

**Pipeline:**

```
PDF statement → text extraction (pdfplumber) → NLP preprocessing → rule-based filter
→ ML classifier → aggregation engine → financial summaries + graphs → Gradio dashboard
```

## Repository structure

```
finsight/
├── data/
│   ├── raw/              # original dataset, untouched
│   └── processed/        # cleaned, localised, synthetic-augmented output
├── src/
│   ├── data_prep.py           # loading, cleaning, localisation, synthetic generation
│   ├── pdf_generator.py       # synthetic PDF statement generator
│   ├── pdf_extraction.py      # pdfplumber extraction module
│   ├── nlp_preprocessing.py   # text cleaning, TF-IDF vectorisation
│   ├── rule_based.py          # rule-based categorisation layer
│   ├── ml_classifier.py       # model training + hyperparameter tuning
│   ├── evaluation.py          # cross-validation, F1, confusion matrix, SHAP
│   ├── insights.py            # rule-based financial summaries
│   ├── visualisations.py      # graphing layer
│   └── storage.py             # Excel read/write layer
├── notebooks/
│   ├── eda.ipynb              # descriptive stats, outlier detection, correlation
│   └── model_comparison.ipynb # side-by-side evaluation of candidate models
├── app.py                     # Gradio interface + pipeline integration
├── tests/
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repository:
   ```
   git clone <repo-url>
   cd finsight
   ```
2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the dashboard locally:
   ```
   python app.py
   ```

## Data

The classification model is trained on a public transaction dataset (~258,000 rows) sourced from GitHub, adapted to reflect South African spending patterns — merchant names, currency (ZAR), and category weighting are adjusted to align with the Stats SA Income and Expenditure Survey. No real personal financial data is used at any point.

Spending categories: Groceries, Transport, Utilities, Entertainment, Health, Education, Other Services.

## Model evaluation

Model performance is assessed primarily using **Macro F1-score**, since category classes are imbalanced and overall accuracy would be misleading. Weighted F1-score, per-class recall, and confusion matrices are used as supporting metrics. Candidate models: Logistic Regression, Naive Bayes, Decision Tree, Random Forest, and XGBoost.

## Workflow

- Each task is tracked as a GitHub Issue and assigned to an owner.
- Work happens on individual branches (`name/task-description`), merged into `main` via Pull Request.
- `main` should always be in a working state.

## Ethical considerations

FINSIGHT does not use real banking data at any stage. All data is either publicly available or synthetically generated. The dashboard is deployed as a private Hugging Face Space, accessible only to the project team and supervisor.

## License

Academic project — Eduvos, Information Technology faculty. Not for commercial use.
