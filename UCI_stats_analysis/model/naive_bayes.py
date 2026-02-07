import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from utils import evaluate_model

def run_naive_bayes(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GaussianNB()
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    return model, metrics