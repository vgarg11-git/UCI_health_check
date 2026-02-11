# ML Assignment 2 - Classification Models

## Problem Statement
To build a predictive model that uses patient medical attributes to determine whether heart disease is present, thereby supporting early diagnosis and preventive healthcareImplement and compare multiple classification models on a chosen dataset, deployed via Streamlit Cloud.

## Dataset
Chosen dataset: UCI Heart Disease Dataset
- Features: ≥ 12
- Instances: ≥ 500

## Models Implemented
1. Logistic Regression
2. Decision Tree
3. kNN
4. Naive Bayes
5. Random Forest
6. XGBoost

## Evaluation Metrics
| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|-------|----------|-----|-----------|--------|----|-----|
| Logistic Regression |   |   |   |   |   |   |
| Decision Tree       |   |   |   |   |   |   |
| kNN                 |   |   |   |   |   |   |
| Naive Bayes         |   |   |   |   |   |   |
| Random Forest       |   |   |   |   |   |   |
| XGBoost             |   |   |   |   |   |   |

## Observations
| Model | Observation |
|-------|-------------|
| Logistic Regression | Performs well with linear seperability, stable metrics |
| Decision Tree       | slightly lower accuracy, prone to overfitting |
| kNN                 | Balanced but sensitive to choice of k and scaling |
| Naive Bayes         | fast and simple, assumes feature indepedence |
| Random Forest       | strong ensemble, robust against overfit, better accuracy |
| XGBoost             | Overall better performance, handles feature interaction well |

## Deployment
- Deployed on Streamlit Community Cloud
- [Live App Link](https://share.streamlit.io/UCI_health_check/UCI_stats_analysis/app.py)
- https://ucihealthcheck-vgarg11.streamlit.app/
