import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from utils import evaluate_model, preprocess

def run_logistic_regression(df, target):
    X, y = preprocess(df, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    return model, metrics, X_test, y_test

