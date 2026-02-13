# Telecom Customer Churn Prediction

Predicting customer churn for a telecom service company using machine learning on Databricks. The dataset is sourced from Kaggle.

## What This Project Does

Customer churn — when subscribers stop using a telecom provider's services — is one of the most expensive problems in the industry. This project builds a predictive pipeline that identifies customers likely to churn based on their usage patterns, account details, and service subscriptions. The goal is to flag at-risk customers early so retention teams can step in before it's too late.

## Project Structure

```
telecom_churn/
├── data/
│   ├── config.py        # Data configuration and path settings
│   ├── evaluate.py      # Data-level evaluation utilities
│   └── predict.py       # Data processing for prediction inputs
├── models/
│   ├── dataset.py       # Dataset loading and preparation
│   ├── model.py         # Model architecture and training logic
│   ├── predict.py       # Inference and prediction pipeline
│   └── utils.py         # Model helper functions
├── src/
│   ├── config.py        # Global configuration
│   ├── dataset.py       # Core dataset handling
│   ├── evaluate.py      # Model evaluation metrics
│   ├── model.py         # Main model definitions
│   └── utils.py         # Shared utility functions
├── config.yaml          # YAML-based project configuration
├── requirements.txt     # Python dependencies
└── README.md
```

## Tech Stack

- **Python** — all source code
- **Databricks** — used as the compute and notebook environment for training and experimentation
- **NumPy** — numerical operations
- **pytest** — testing

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/hirthickraj2015/telecom_churn.git
   cd telecom_churn
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Update `config.yaml` with your dataset paths and any environment-specific settings.

4. If running on Databricks, import the project into your workspace and configure your cluster with the required packages.

## Dataset

The dataset comes from Kaggle and contains telecom customer records with features like account length, call minutes, service plan subscriptions, customer service call counts, and churn labels. You'll need to download it separately from Kaggle and place it in the expected data directory.

## License

No license specified.
