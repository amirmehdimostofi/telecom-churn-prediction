# Telecom churn prediction

Predict which customers are likely to leave a telecom provider, using call-detail and CRM data.

## Data

`cs_06.csv` has 3,150 customers from an Iranian telecom company, recorded over 12 months (September 2006–September 2007). Features include call failures, complaints, subscription length, charges, call and SMS activity, age group, tariff plan, and account status. About 16% of customers in the file are marked as churned (`if_churn`).

## Method

The notebook (`classification.ipynb`) cleans and explores the data, then compares classification models in Python with pandas and scikit-learn:

- Logistic regression (including class weights, oversampling, and undersampling)
- k-nearest neighbours
- Random forest
- Stochastic gradient boosting
- Support vector machine

Models are evaluated with accuracy, precision, recall, F1, and F2.

## Findings

From the comparison table in the notebook, a tuned SVM with oversampling was the strongest model: accuracy 0.961, precision 0.819, recall 0.967, F1 0.887, and F2 0.933. A class-weighted, tuned random forest was next (accuracy 0.947, F1 0.842). These figures are from the notebook as committed and have not been re-run independently.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook classification.ipynb
```
