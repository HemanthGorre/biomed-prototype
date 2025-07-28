import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import streamlit as st

def is_classification(df, y_col):
    y = df[y_col]
    return pd.api.types.is_integer_dtype(y) and y.nunique() <= 10

def get_models(task_type):
    if task_type == 'regression':
        return {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(),
            "XGBoost": XGBRegressor(objective='reg:squarederror', verbosity=0)
        }
    else:
        return {
            "Logistic Regression": LogisticRegression(max_iter=200),
            "Random Forest": RandomForestClassifier(),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }

def train_and_evaluate(df, x_cols, y_col, model_name, normalization, test_size=0.2, random_state=42):
    X = df[x_cols]
    y = df[y_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    task_type = 'classification' if is_classification(df, y_col) else 'regression'
    models = get_models(task_type)
    model = models[model_name]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {}

    if task_type == 'regression':
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        metrics = {"R²": r2, "RMSE": rmse}
    else:
        acc = accuracy_score(y_test, y_pred)
        if y_test.nunique() == 2:
            probas = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probas)
            metrics = {"Accuracy": acc, "ROC-AUC": auc}
        else:
            metrics = {"Accuracy": acc}
    # Added: also return model object and feature names
    return metrics, (model, X_test, y_test, y_pred, task_type, x_cols)

def compare_models_and_preprocessing(df, x_cols, y_col):
    results = []
    for normalization in ['StandardScaler', 'Min-Max Scaler', 'RobustScaler']:
        from .preprocessing import preprocess_data
        X_scaled = preprocess_data(df, x_cols, normalization)
        temp_df = df.copy()
        temp_df[x_cols] = X_scaled
        task_type = 'classification' if is_classification(df, y_col) else 'regression'
        models = get_models(task_type)
        for model_name in models.keys():
            metrics, _ = train_and_evaluate(temp_df, x_cols, y_col, model_name, normalization)
            res = {
                "Normalization": normalization,
                "Model": model_name,
                **metrics
            }
            results.append(res)
    return pd.DataFrame(results)

def plot_roc_curve(model, X_test, y_test):
    probas = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probas)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label='ROC curve')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)
