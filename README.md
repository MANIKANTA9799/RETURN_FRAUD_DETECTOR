
1. Project Overview

This project builds a complete machine-learning solution to detect fraudulent e-commerce transactions.
A full pipeline was developed including data preprocessing, feature engineering, model training, model comparison, threshold tuning, and deployment-ready saving of the final system.
XGBoost with optimized decision thresholding produced the best overall fraud-detection performance.

2. Objective

The goal is to assign each transaction a fraud probability and use an optimized threshold to classify it as fraudulent or legitimate.
Since fraud is extremely rare and costly, the project emphasizes precision–recall tradeoffs, PR-AUC, and threshold optimization rather than accuracy.

3. Dataset Description

The dataset contains ~273k e-commerce transactions with both raw and engineered features.
Key variables include:

Transaction Amount

Payment Method

Product Category

Customer Age

Customer Location

Device Used

IP Address

Account Age Days

Transaction Hour

Shipping/Billing Address

Is Fraudulent (target)

Additional engineered features include:

Location-based fraud rate

IP usage count

IP fraud history

Weekend/weekday features

Address mismatch indicator

4. Data Preprocessing & Feature Engineering

Major preprocessing steps:

Time-Based Features

Extracted Month, Weekday, IsWeekend

Removed Day and Year (non-informative)

Converted timestamps to proper datetime

Risk & Behavior Features

Loc_Fraud_Rate = fraud rate by customer location

IP_Fraud_History = fraud rate per IP

IP_Usage_Count = transactions per IP

Address_Mismatch = shipping vs billing mismatch

Categorical Encoding

One-hot encoding for Payment Method, Product Category, Device Used

Numerical Scaling

StandardScaler applied to amount, age, hour, and other continuous variables

Final Cleanup

Removed Transaction ID

Removed Customer ID after extracting behavior features

Prepared an optimized balanced dataset for model experimentation

5. Modeling Approach

Models evaluated:

Logistic Regression

Random Forest

XGBoost

Threshold-tuned XGBoost

Metrics used:

Precision

Recall

F1 Score

PR-AUC (primary metric for imbalanced data)

ROC-AUC

Confusion matrix

Learning curves

Precision-recall curves


6. Why XGBoost Was Selected

Models complex nonlinear fraud patterns

Achieves highest PR-AUC (most important fraud metric)

Best tradeoff of precision and recall

Threshold tuning improves F1 significantly

Stable probability separation between fraud and non-fraud

8. Threshold Optimization

A probability sweep from 0.01 → 0.99 was performed.
F1 score peaked at:

BEST_THRESHOLD = 0.66623265


This threshold yields the best balance between false positives and false negatives.
