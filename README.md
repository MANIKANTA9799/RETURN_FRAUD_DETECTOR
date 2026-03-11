# Return Fraud Detection System (Machine Learning)
### Live Demo
[Open the deployed fraud detection app](https://returnfrauddetector-mdclrvyva4c5p277zx8u6s.streamlit.app/)

## Overview
Fraudulent transactions represent a major financial risk for e-commerce platforms. Detecting fraud is challenging because fraudulent transactions are rare, complex, and often hidden among millions of legitimate purchases. This project builds a complete machine learning pipeline for detecting fraudulent e-commerce transactions using behavioral, transactional, and contextual features derived from raw transaction data.

The system covers the full ML workflow including data preprocessing, feature engineering, model training, evaluation, threshold optimization, and model export. Multiple models were evaluated including Logistic Regression, Random Forest, and XGBoost. The final model was selected using metrics suitable for imbalanced datasets, particularly Precision–Recall AUC and F1 Score.

The final system demonstrates how machine learning can capture complex fraud patterns and provide a practical operating threshold for real-world fraud detection systems.

---

## Problem Statement
Fraud detection is a binary classification problem where the goal is to determine whether a transaction is fraudulent or legitimate.

The dataset is highly imbalanced because fraudulent transactions occur far less frequently than legitimate ones. Because of this imbalance, traditional metrics such as accuracy are misleading.

The main objective of this project is to build a model that can reliably detect fraudulent transactions while balancing:

Precision — minimizing false fraud alerts  
Recall — capturing as many fraud cases as possible

---

## Dataset
Two publicly available datasets were initially explored:

1. Fraudulent E-Commerce Transaction Dataset  
2. E-commerce Returns Dataset

The goal was to combine both datasets to obtain richer information, since the returns dataset contains useful behavioral attributes while the fraud dataset includes the fraud label. However, after inspecting the customer identifiers, the IDs between the two datasets did not match, making a reliable merge impossible.

Therefore, the fraud transaction dataset was used as the primary dataset.

The dataset contains transaction-level attributes such as:

Transaction amount  
Customer age  
Account age  
Device used  
Payment method  
Product category  
IP address  
Customer location  
Shipping address  
Billing address  
Transaction timestamp  

Target variable:

Is Fraudulent (1 = Fraud, 0 = Legitimate)

---

## Feature Engineering
Several domain-inspired features were engineered to capture fraud behavior patterns.

### Location-Based Features
Customer location was converted into a fraud-rate feature:

Loc_Fraud_Rate = average fraud rate of transactions from that location

This helps capture geographic patterns of fraud activity.

### IP Behavior Features
Fraud often involves suspicious IP usage patterns. The following features were created:

Customer_IP_Count — number of unique IPs used by a customer  
IP_Usage_Count — number of transactions associated with an IP  
IP_Fraud_History — historical fraud rate associated with an IP

These features help detect IP sharing and abnormal usage patterns.

### Time-Based Features
Transaction timestamps were transformed into structured time features:

Month  
Weekday  
Transaction Hour  
Weekend indicator

Fraud patterns frequently correlate with unusual transaction timing.

### Customer History Features
Historical customer behavior was captured using cumulative statistics:

Cust_Past_Fraud  
Cust_Fraud_Flag  
Cust_Txn_Count  
Cust_Fraud_Rate

These features represent the historical fraud behavior of each customer. If these features were found to contain no useful signal, they were automatically removed.

### Address Mismatch Feature
Fraudulent orders often have mismatched billing and shipping addresses.

Address_Mismatch = 1 if Shipping Address ≠ Billing Address

### Categorical Encoding
Nominal categorical variables were encoded using one-hot encoding:

Payment Method  
Product Category  
Device Used

### Feature Scaling
Continuous numerical variables were standardized using StandardScaler to ensure consistent numerical ranges.

---

## Handling Class Imbalance
The original dataset contains significantly more legitimate transactions than fraudulent ones. Training on the full dataset locally would be computationally expensive and would bias the model toward the majority class.

To address this:

All fraud samples were retained  
A subset of non-fraud samples was randomly selected

This produced a balanced training dataset while preserving fraud diversity.

---

## Model Training

### Logistic Regression
Logistic Regression was used as a baseline model.

Findings:
- Unable to model complex nonlinear fraud patterns
- Precision–Recall curve showed weak separation
- Learning curves indicated high bias and underfitting

Conclusion: Logistic Regression is too simple for this problem.

---

### Random Forest
Random Forest was used to capture nonlinear feature interactions.

Findings:
- Significant improvement over Logistic Regression
- Higher precision indicating fewer false positives
- Moderate recall indicating some fraud cases were still missed

However, the model showed signs of overfitting and limited generalization.

---

### XGBoost
XGBoost was used as the final model because of its ability to learn complex feature interactions using gradient boosting.

Advantages:
- Sequential tree learning that corrects previous errors
- Effective modeling of nonlinear patterns
- Built-in regularization to reduce overfitting

Results showed that XGBoost provided the best balance between fraud detection capability and prediction reliability.

---

## Model Evaluation Metrics
Because fraud datasets are highly imbalanced, accuracy is not a reliable metric. Instead, the following metrics were used.

Precision  
Measures the percentage of predicted fraud transactions that are actually fraud.

Recall  
Measures how many fraudulent transactions the model successfully detected.

F1 Score  
Harmonic mean of precision and recall, ensuring balanced performance.

ROC-AUC  
Measures ranking quality of predictions.

PR-AUC (Average Precision)  
Primary evaluation metric for imbalanced datasets. It measures how well the model separates fraud from legitimate transactions.

---

## Threshold Optimization
Machine learning models output probabilities rather than direct classifications. The default threshold of 0.5 is often suboptimal for fraud detection.

To determine the best operating threshold:

1. Precision–Recall vs Threshold curves were analyzed.
2. F1 scores were calculated across thresholds from 0.01 to 0.99.
3. The threshold that maximized the F1 score was selected.

The optimal threshold for the XGBoost model was approximately:

Threshold = 0.666

This threshold produced the best balance between detecting fraud and avoiding excessive false alarms.

---

## Model Comparison Summary

Logistic Regression  
Underfitting due to inability to capture nonlinear fraud patterns.

Random Forest  
High precision but moderate recall, missing several fraud cases.

XGBoost  
Best overall performance with the highest PR-AUC and strongest balance between precision and recall.

Therefore, XGBoost was selected as the final model.

---

## Final Model
The trained XGBoost model was exported using joblib for deployment or inference.

Saved model file:

xgb_fraud_model.pkl

---



## Technologies Used
Python  
NumPy  
Pandas  
Scikit-learn  
XGBoost  
Matplotlib  
Seaborn  
Joblib

---

## Future Improvements
Several enhancements could further improve the system:

Hyperparameter optimization using Bayesian search  
Cross-validation with TimeSeriesSplit  
Feature importance analysis for interpretability  
Real-time fraud detection pipeline  
Deployment as a REST API  
Integration with anomaly detection models

---

## Conclusion
This project demonstrates a complete fraud detection machine learning pipeline including feature engineering, model evaluation, and threshold optimization. Through systematic experimentation with multiple algorithms, XGBoost was identified as the most effective model for capturing complex fraud behavior patterns. The final model achieves a strong balance between precision and recall and provides a practical decision threshold suitable for real-world fraud detection systems.
