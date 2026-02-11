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
| Logistic Regression | 0.57  |0.50   | 0.57  | 0.53  |0.37   |  NULL |
| Decision Tree       | 0.59  |0.52   | 0.59  | 0.54  |0.40   | NULL  |
| kNN                 | 0.53  |0.46   | 0.53  | 0.47  |0.30   | NULL  |
| Naive Bayes         | 0.47  |0.69   | 0.47  | 0.51  |0.36   | NULL  |
| Random Forest       | 0.61  |0.57   | 0.61  | 0.58  |0.43   | NULL  |
| XGBoost             | 0.60  |0.57   | 0.60  | 0.58  |0.42   | NULL  |

## Observations
| Model | Observation |
|-------|-------------|
| Logistic Regression | Performs well with linear seperability, stable metrics |
| Decision Tree       | slightly better accuracy, prone to overfitting |
| kNN                 | Balanced but sensitive to choice of k and scaling |
| Naive Bayes         | fast and simple, assumes feature indepedence |
| Random Forest       | strong ensemble, robust against overfit, better accuracy |
| XGBoost             | Overall better performance, handles feature interaction well |

## Deployment
- Deployed on Streamlit Community Cloud
- [Live App Link](https://share.streamlit.io/UCI_health_check/UCI_stats_analysis/app.py)
- https://ucihealthcheck-vgarg11.streamlit.app/
