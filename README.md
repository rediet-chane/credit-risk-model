# Credit Risk Probability Model for Alternative Data

## Project Overview
This project builds a credit scoring model for Bati Bank to enable buy-now-pay-later services. The model uses transaction data from an eCommerce partner to predict customer credit risk.

## Credit Scoring Business Understanding

### Basel II Compliance
The Basel II Capital Accord emphasizes risk measurement, interpretability, and documentation. In a regulated financial context, models must be explainable to auditors and regulators. This influences the choice of interpretable models (like Logistic Regression with Weight of Evidence) over black-box models when regulatory compliance is a priority.

### Proxy Variable Necessity
The dataset contains no direct default label. Therefore, a proxy target variable is necessary. Using RFM (Recency, Frequency, Monetary) analysis, customers are segmented into risk groups. However, this introduces business risks: proxy-based prediction may misclassify customers who are genuinely low-risk but have low transaction activity.

### Model Trade-offs
| Model Type | Pros | Cons |
|-------------|------|------|
| Logistic Regression (WoE) | Interpretable, regulator-friendly | Lower predictive power |
| Gradient Boosting | Higher accuracy | Black box, harder to explain |

In a regulated financial context, the trade-off is between transparency (Basel II compliance) and predictive performance. The optimal approach is to use both: an interpretable model for regulatory reporting and a high-performance model for internal risk assessment.

## Data Source
Xente Challenge Dataset (Kaggle)

## Setup Instructions
1. Create virtual environment: `python -m venv venv`
2. Activate: `venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
4. Run Jupyter: `jupyter notebook notebooks/eda.ipynb`

## Project Structure