# Telco Customer Churn Prediction

Predicting churn with XGBoost and model explainability techniques (PDP + SHAP)

## Overview

This project focuses on predicting customer churn for a telecommunications operator using machine learning.
Identifying customers likely to churn allows businesses to make targeted retention programs, with the goal of reducing revenue loss and improving customer satisfaction.

Beyond model performance, a key goal of this project is to understand **why** the model makes its predictions.
For this reason, the notebook includes model transparency and explainability through Partial Dependence Plots (PDP) and SHAP values.

## Dataset

The project uses the Telco Customer Churn dataset from Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The dataset contains customer information:
* **Churn label**: whether the customer left in the last month 
* **Demographics**: gender, seniority, partner/dependent status 
* **Account details**: tenure, contract type, payment method, monthly and total charges 
* **Service subscriptions**: phone lines, internet service, streaming services, technical support, etc.

## Preprocessing

The data required several cleaning and preparation steps for compatibility with the model:
* Handled missing values in the `TotalCharges` column (caused by tenure = 0 customers)
* Manually encoded categorical features for improved interpretability
* Split the dataset with stratification to preserve class distribution


## Exploratory Data Analysis (EDA):
A series of visualizations were created to understand patterns in the features:
* Class balance inspection
* Distribution plots for key numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`)
* Churn rate across contract types, payment methods, and services
* Numerical correlation between Churn and the other features

EDA provided the foundation for understanding which customer segments are most vulnerable to churn.

## Modeling

The prediction model used is a tuned XGBoost, a gradient boosting algorithm known for its strong performance on tabular data.
Threshold tuning was performed to find a more business-appropriate decision boundary (lower than the default 0.5).

## Explainability

A key part of the project is understanding why the model makes its predictions.
Interpretability is crucial in churn analysis, preventing the model from becoming a black box.

This project incorporates both global and local explainability methods:
* **Feature importances**: XGBoost's built-in feature importances show the most influential features
* **Partial Dependence Plots**: these provide intuition for the direction and shape of the relationship between a feature and predicted churn probability
* **SHAP force plot** for a single prediction: SHAP values reveal which features pushed the model towards predicting churn/non-churn for a single example
* **SHAP summary plot**: it shows whether low or high values for every feature push towards churn, and how strongly they do so.

## Conclusion

This project successfully builds a churn prediction model and critically explains why customers are likely to churn.
The combined use of EDA, XGBoost, PDPs, and SHAP provides both predictive power and interpretability, making the results usable for real-world retention strategies.
