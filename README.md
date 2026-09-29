Credit Card Fraud Detection

Machine learning project to detect fraudulent credit card transactions in a highly imbalanced dataset, using XGBoost with class-weighting, hyperparameter tuning, and SHAP-based explainability.

Problem

Fraudulent transactions make up only 0.17% of all transactions (492 out of 284,807). A naive model that predicts "not fraud" for everything would score 99.8% accuracy while catching zero fraud cases — so this project focuses on metrics that actually matter for imbalanced classification: Precision, Recall, F1-score, and ROC-AUC.

Dataset
Source: Credit Card Fraud Detection — Kaggle
Size: 284,807 transactions, 31 columns
Features: V1–V28 (PCA-anonymized for privacy), Time, Amount
Target: Class (0 = normal, 1 = fraud)
No missing values.
Approach
EDA — explored the class imbalance and feature distributions.
Preprocessing — scaled Amount with StandardScaler, stratified train/test split (80/20) to preserve the fraud ratio in both sets.
Baseline model — Logistic Regression, used as a reference point.
Main model — XGBoost with scale_pos_weight to handle class imbalance.
Hyperparameter tuning — GridSearchCV over max_depth, learning_rate, and n_estimators.
Explainability — SHAP values to understand which features drive each prediction.
Serving — a simple FastAPI endpoint that returns a fraud probability for a given transaction.
Results
Model	Precision	Recall	F1-score	ROC-AUC
Logistic Regression (baseline)	0.86	0.58	0.70	~0.90
XGBoost (default)	0.98	0.83	0.90	0.981
XGBoost (tuned)	0.98	0.82	0.89	0.982

