import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv")
y_test = pd.read_csv("data/processed/y_test.csv")
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()
print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_log)

print("Logistic Regression Accuracy:", accuracy)

print(classification_report(y_test, y_pred_log))

print(confusion_matrix(y_test, y_pred_log))

roc_score = roc_auc_score(y_test, y_pred_log)

print("ROC-AUC Score:", roc_score)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, y_pred_rf)

print("Random Forest Accuracy:", rf_accuracy)

print(classification_report(y_test, y_pred_rf))


print(confusion_matrix(y_test, y_pred_rf))

rf_roc = roc_auc_score(y_test, y_pred_rf)

print("Random Forest ROC-AUC:", rf_roc)

xgb_model = XGBClassifier(
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

xgb_accuracy = accuracy_score(y_test, y_pred_xgb)

print("XGBoost Accuracy:", xgb_accuracy)

print(classification_report(y_test, y_pred_xgb))

print(confusion_matrix(y_test, y_pred_xgb))

xgb_roc = roc_auc_score(y_test, y_pred_xgb)

print("XGBoost ROC-AUC:", xgb_roc)

print("\nMODEL COMPARISON")

print(f"Logistic Regression Accuracy: {accuracy:.4f}")
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")

print()

print(f"Logistic Regression ROC-AUC: {roc_score:.4f}")
print(f"Random Forest ROC-AUC: {rf_roc:.4f}")
print(f"XGBoost ROC-AUC: {xgb_roc:.4f}")

joblib.dump(
    log_model,
    "models/logistic_regression_model.pkl"
)

print("Best model saved successfully!")

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": abs(log_model.coef_[0])
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance.head(10))

top_features = feature_importance.head(10)

plt.figure(figsize=(10,6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.title("Top Features Influencing Customer Churn")

plt.gca().invert_yaxis()

plt.show()

'''explainer = shap.LinearExplainer(
    log_model,
    X_train
)

shap_values = explainer.shap_values(X_test)

shap.summary_plot(
    shap_values,
    X_test,
    show=False
)

plt.show()'''

explainer = shap.LinearExplainer(
    log_model,
    X_train
)

shap_values = explainer.shap_values(X_test)

print("SHAP values generated successfully!")
print(shap_values[:5])