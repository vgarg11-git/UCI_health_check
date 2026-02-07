import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from utils import evaluate_model, preprocess

def run_decision_tree(df, target):
   # One-hot encode categorical features
    #X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
    #y = df[target]
    X, y = preprocess(df, target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    return model, metrics



