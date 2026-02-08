import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from model import logistic_regression, decision_tree, knn, naive_bayes, random_forest, xgboost_model
from utils import preprocess

st.title("ML Assignment 2 - Classification Models")

# a. Dataset upload option
uploaded_file = st.file_uploader("Upload CSV dataset", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Select target column
    target = st.selectbox("Select Target Column", df.columns)

    # b. Model selection dropdown
    model_choice = st.selectbox("Choose Model", 
        ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest", "XGBoost"])

    if st.button("Run Model"):
        if model_choice == "Logistic Regression":
            model, metrics = logistic_regression.run_logistic_regression(df, target)
        elif model_choice == "Decision Tree":
            model, metrics, X_test, y_test = decision_tree.run_decision_tree(df, target)
        elif model_choice == "kNN":
            model, metrics = knn.run_knn(df, target)
        elif model_choice == "Naive Bayes":
            model, metrics = naive_bayes.run_naive_bayes(df, target)
        elif model_choice == "Random Forest":
            model, metrics = random_forest.run_random_forest(df, target)
        elif model_choice == "XGBoost":
            model, metrics = xgboost_model.run_xgboost(df, target)

        # c. Display evaluation metrics
        st.write("### Evaluation Metrics")
        st.json(metrics)

        # d. Confusion matrix and classification report
        st.write("### Confusion Matrix")
        X = df.drop(columns=[target])
        y = df[target]
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)


        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.write("### Classification Report")
        report = classification_report(y, y_pred, output_dict=False)

        st.text(report)





