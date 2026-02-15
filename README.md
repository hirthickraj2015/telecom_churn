# Telecom Customer Churn Prediction

A multi-environment Databricks ML pipeline for predicting customer churn in telecom services. The project follows the medallion architecture (raw → bronze → silver) and uses Logistic Regression for binary classification.

## What This Project Does

Customer churn — when subscribers stop using a telecom provider's services — is one of the most expensive problems in the industry. This project builds a predictive pipeline that identifies customers likely to churn based on their usage patterns, account details, and service subscriptions. The goal is to flag at-risk customers early so retention teams can step in before it's too late.

## Project Structure

```
telecom_churn/
├── notebook/
│   ├── dev/                       # Development environment
│   ├── prod/                      # Production environment
│   └── uat/                       # User Acceptance Testing environment
│       └── [each environment contains]:
│           ├── nb_raw/
│           │   └── nb_setup.ipynb           # Raw data ingestion setup
│           ├── nb_bronze/
│           │   └── nb_incremental_load.ipynb # Incremental data loading
│           └── nb_silver/
│               └── nb_silver.ipynb          # Data transformation & ML training
├── workflows/
│   ├── dev-ci.yml                 # Development CI pipeline
│   └── prod-ci.yml                # Production CI pipeline
├── LICENSE
└── README.md
```

## Data Pipeline (Medallion Architecture)

| Layer | Notebook | Purpose |
|-------|----------|---------|
| **Raw** | `nb_setup.ipynb` | Creates Databricks volumes and directory structure for raw data |
| **Bronze** | `nb_incremental_load.ipynb` | Streams CSV data from raw volumes into Delta Lake tables |
| **Silver** | `nb_silver.ipynb` | Data transformation, feature engineering, model training & evaluation |

## ML Model

- **Algorithm:** Logistic Regression (scikit-learn)
- **Target:** Customer churn (binary classification: Yes/No)
- **Training:** 80/20 train-test split
- **Evaluation Metrics:** Accuracy, F1 Score, Confusion Matrix, Classification Report

## Tech Stack

- **Databricks** — compute and notebook environment
- **PySpark / Spark SQL** — distributed data processing
- **Delta Lake** — ACID-compliant table format
- **Pandas** — DataFrame operations
- **scikit-learn** — ML model training and evaluation
- **Matplotlib / Seaborn** — data visualization
- **GitHub Actions** — CI/CD automation

## Dataset Features

**Input Features (from Kaggle telecom dataset):**
- Demographics: gender, senior citizen status, partner, dependents
- Account: tenure, phone service, paperless billing
- Services: multiple lines, internet service, online security, online backup, device protection, tech support, streaming TV, streaming movies
- Contract: contract type (Month-to-month, One year, Two year)
- Billing: payment method, monthly charges, total charges

**Target Variable:** Churn (Yes/No)

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/hirthickraj2015/telecom_churn.git
   cd telecom_churn
   ```

2. Import the project into your Databricks workspace.

3. Configure your Databricks cluster with the required packages (PySpark, pandas, scikit-learn, matplotlib, seaborn).

4. Download the telecom customer churn dataset from Kaggle and upload it to your Databricks volume.

5. Run the notebooks in order: `nb_setup.ipynb` → `nb_incremental_load.ipynb` → `nb_silver.ipynb`

## CI/CD

The project includes GitHub Actions workflows for automated deployments:
- `dev-ci.yml` — triggers on pushes to the develop branch
- `prod-ci.yml` — triggers on pushes to the main branch

## License

See [LICENSE](LICENSE) file.
