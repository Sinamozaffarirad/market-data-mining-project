# Market Data Mining Project

Django application for analysing the Dunnhumby retail dataset. It includes
association rules, RFM customer segments, customer churn experiments, purchase
recommendations, product revenue forecasting, and a business intelligence dashboard.

## Contents

- `Website/market/`: Django applications, templates, static assets and tests.
- `Website/market/ml_models_cache/`: saved models and recommendation matrices.
- `database/`: a full backup of the `marketdb` database, taken from the live
  database on 8 September 2026 and verified with `RESTORE VERIFYONLY`.
- `dashboardMarket.pbix`: Power BI report.
- `report/Project_Report.docx` and `report/Project_Report.pdf`: final written report.

## Local setup on Windows

The supplied environment was checked with Python 3.14.6 and the package versions
in `requirements.txt`. The pinned dependencies require Python 3.11 or newer.
SQL Server and Microsoft ODBC Driver 17 for SQL Server are required separately.

From this folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:DJANGO_SECRET_KEY = & .\.venv\Scripts\python.exe -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

Keep the secret key stable across runs if login sessions need to survive a server
restart. When no key is supplied, the application generates a temporary key for
that process.

Restore `database/marketdb_backup_20260908_222737.bak` in SQL Server Management
Studio as `marketdb`. Use a separate SQL Server instance or a new database name
when an existing database must be preserved. The backup includes data; the CSV
source files are not included. Database restoration is required because several
dataset tables are unmanaged Django models and are not created by migrations.

The default connection uses Windows authentication against `localhost`. Set these
variables if your SQL Server instance, database name or installed driver differs:

```powershell
$env:DB_HOST = 'localhost'
$env:DB_NAME = 'marketdb'
$env:DB_DRIVER = 'ODBC Driver 17 for SQL Server'
.\.venv\Scripts\python.exe Website\market\manage.py migrate
.\.venv\Scripts\python.exe Website\market\manage.py check
.\.venv\Scripts\python.exe Website\market\manage.py runserver
```

Use an existing account from the restored database, or create a local account with
`python manage.py createsuperuser` from `Website/market/` using the virtual
environment's Python interpreter.

Open `http://127.0.0.1:8000/analysis/`. The main pages are:

| Path | Purpose |
| --- | --- |
| `/analysis/basket-analysis/` | Association rules, repurchase models and revenue forecasts |
| `/analysis/customer-segments/` | RFM segments and churn experiments |
| `/analysis/bi-dashboard/` | Retail performance and statistical comparisons |
| `/customers/search/` | Customer search and history |
| `/analysis/product-recommender/` | Recommendation tools |
| `/analysis/customer-retention/` | Retention overview |
| `/analysis/data-management/` | Dataset browsing and maintenance |
| `/admin/` | Django administration |

The styles and charts use external CDN libraries, so the browser needs internet
access. The embedded Power BI report also needs access to its configured workspace.
Override `POWER_BI_EMBED_URL` for another published report. Opening the PBIX file
separately requires Power BI Desktop and may require reconnecting its data source.

This configuration is for a local demonstration. `DJANGO_DEBUG` defaults to `1`;
`DJANGO_ALLOWED_HOSTS` defaults to loopback hosts. Server deployment requires its
own HTTPS, cookie and database configuration.

## Analysis methods

- Association rules: Apriori over baskets at product, commodity and department
  level, reported with support, confidence and lift.
- Customer segmentation: RFM scoring into eleven segments, shared by the
  segmentation dashboard and the churn feature builder.
- Churn: XGBoost over leakage-safe observation windows, with held-out evaluation.
- Repurchase: neural network, random forest, gradient boosting and SVM compared
  on a purged chronological holdout.
- Revenue forecasting: a recursive network and an independent direct model,
  both measured against a recent-average benchmark.
- Recommendations: collaborative filtering combined with a trained model and
  the stored association rules.
- Group comparisons: Mann-Whitney U, Kolmogorov-Smirnov, chi-square, Welch's
  t-test, one-way ANOVA and Kruskal-Wallis, each reported with an effect size
  (Cliff's delta, Cramer's V, Cohen's d, eta squared, epsilon squared) and with
  p-values adjusted for multiple comparisons by the Benjamini-Hochberg
  procedure. Groups can be compared on basket value or on visits per household.

## Models and checks

The current loaders use version 12 revenue forecasts, version 4 household-department
repurchase models, and version 1 hybrid recommendation models. Keep the supplied
model directories in place to reuse trained results. The collaborative-filtering
cache is large because it stores the household-product matrix and similarities.

Forecasts use ordered 30-day periods. Available revenue forecast horizons are
1–9 periods, lookback windows are 2–12 periods, and sliding steps are 1–3 periods.
The application validates whether each requested combination has enough history.
Training and validation use chronological boundaries to avoid future-data leakage.

Run the numerical and model-behaviour tests without creating a database:

```powershell
.\.venv\Scripts\python.exe Website\market\manage.py test dunnhumby
```

The customer integration test requires an isolated test database. Database
maintenance and training commands change data or model artifacts.

Additional commands, from `Website/market/` with the virtual environment active:

```powershell
python manage.py build_cf_cache
python manage.py train_hybrid_recommender --help
python manage.py import_transactions --help
```

The transaction importer is a legacy dataset-specific utility and requires an
explicit `--csv-path`. Check its column mapping against your CSV before using it;
restoring the supplied backup is the documented route for the demonstration.

## Project information

Sina Mozaffarirad, Fatemeh Dastyafteh and Samaneh Tabandeh — university market
data mining project.

Dataset: Dunnhumby, The Complete Journey. The application uses Django,
scikit-learn, XGBoost, pandas, NumPy, SciPy, Bootstrap and Chart.js.
Third-party tools and dataset materials retain their respective terms.
