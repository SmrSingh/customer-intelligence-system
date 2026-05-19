import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
model = joblib.load(
    "models/churn_pipeline.pkl"
)


st.title("Customer Churn Intelligence System")

st.write(
    "Predict customer churn using machine learning."
)

st.header("Customer Information")

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior_citizen = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

tenure = st.slider(
    "Tenure (Months)",
    0,
    72,
    12
)

monthly_charges = st.slider(
    "Monthly Charges",
    0.0,
    150.0,
    70.0
)

contract = st.selectbox(
    "Contract Type",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)
internet_service = st.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)

tech_support = st.selectbox(
    "Tech Support",
    [
        "Yes",
        "No"
    ]
)

online_security = st.selectbox(
    "Online Security",
    [
        "Yes",
        "No"
    ]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)
partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

phone_service = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["Yes", "No"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["Yes", "No"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No"]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=100.0
)

st.write("Customer Input Summary")

st.write({
    "Gender": gender,
    "SeniorCitizen": senior_citizen,
    "Tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "Contract": contract
})
if st.button("Predict Churn"):

    input_data = {
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "Contract": contract,
    "InternetService": internet_service,
    "TechSupport": tech_support,
    "OnlineSecurity": online_security,
    "PaymentMethod": payment_method,
    "Partner": partner,
    "Dependents": dependents,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "PaperlessBilling": paperless_billing,
    "TotalCharges": total_charges,
}

    input_df = pd.DataFrame([input_data])
    


 

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(
    input_df
    )[0][1]

if prediction == 1:

      st.error(
        f"Customer likely to churn ({probability*100:.1f}%)"
      )

      st.progress(float(probability))
      st.warning("Suggested Retention Actions")

      if contract == "Month-to-month":
        st.write("- Offer yearly discount plan")

      if tech_support == "No":
        st.write("- Provide free tech support")

      if online_security == "No":
        st.write("- Offer security add-on")

      if monthly_charges > 80:
        st.write("- Recommend lower-cost plan")

else:

      st.success(
        f"Customer likely to stay ({(1 - probability)*100:.1f}%)"
      )

      st.progress(float(1 - probability))


    

      

fig = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={"text": "Churn Risk"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "red"},
            "steps": [
                {"range": [0, 40], "color": "green"},
                {"range": [40, 70], "color": "yellow"},
                {"range": [70, 100], "color": "red"}
            ]
        }
    )
)

st.plotly_chart(fig)
st.subheader("Why this prediction?")

reasons = []

if contract == "Month-to-month":
    reasons.append("Month-to-month contracts increase churn risk")

if monthly_charges > 80:
    reasons.append("High monthly charges increase churn probability")

if tech_support == "No":
    reasons.append("Lack of tech support increases churn risk")

if online_security == "No":
    reasons.append("No online security is associated with churn")

if tenure < 12:
    reasons.append("New customers are more likely to churn")

for reason in reasons:
    st.write("•", reason)