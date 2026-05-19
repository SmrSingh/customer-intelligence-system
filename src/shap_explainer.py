import joblib
import pandas as pd
import shap

pipeline = joblib.load(
    "models/churn_pipeline.pkl"
)

df = pd.read_csv(
    "data/raw/telco_churn.csv"
)
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

X = df.drop(
    ["customerID", "Churn"],
    axis=1
)

sample_data = X.sample(50)

transformed_data = pipeline.named_steps[
    "preprocessor"
].transform(sample_data)


feature_names = pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()

explainer = shap.Explainer(
    pipeline.named_steps[
        "classifier"
    ].predict,
    transformed_data
)

shap_values = explainer(
    transformed_data
)



shap.summary_plot(
    shap_values,
    transformed_data,
    feature_names=feature_names
)

print("SHAP summary plot generated!")