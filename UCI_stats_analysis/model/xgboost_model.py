import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from utils import evaluate_model

def run_xgboost(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    return model, metrics