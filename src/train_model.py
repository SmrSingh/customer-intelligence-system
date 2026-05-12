import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

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