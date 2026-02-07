import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = None
    try:
        y_prob = model.predict_proba(X_test)[:,1]
    except:
        pass
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1": f1_score(y_test, y_pred, average="weighted"),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else None
    }

    # Handle AUC for binary vs multi-class
    if y_prob is not None:
        try:
            if len(set(y_test)) > 2:
                metrics["AUC"] = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
            else:
                metrics["AUC"] = roc_auc_score(y_test, y_prob[:,1])
        except:
            metrics["AUC"] = None
    else:
        metrics["AUC"] = None

    return metrics
