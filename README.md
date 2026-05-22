# Customer Churn Prediction System

An end-to-end Machine Learning web application that predicts customer churn using behavioral and subscription-based customer data. The project includes data preprocessing, model training, Docker containerization, and cloud deployment using Render.

## Live Demo

https://customer-churn-app-ouzr.onrender.com

## GitHub Repository

https://github.com/SmrSingh/customer-intelligence-system

---

# Features

- Customer churn prediction using Machine Learning
- Interactive Streamlit dashboard
- Real-time prediction probability scoring
- Customer risk visualization
- Retention recommendation engine
- End-to-end ML pipeline
- Dockerized deployment
- Cloud deployment on Render

---

# Tech Stack

## Machine Learning
- Python
- Scikit-learn
- Pandas
- NumPy

## Visualization & Frontend
- Streamlit
- Plotly

## Deployment
- Docker
- Render
- GitHub

---

# Machine Learning Workflow

1. Data preprocessing and cleaning
2. Feature engineering
3. Model training using Logistic Regression
4. Model evaluation using Accuracy and ROC-AUC
5. Model serialization using Pickle
6. Streamlit app integration
7. Docker containerization
8. Cloud deployment on Render

---

# Model Performance

| Metric | Score |
|---|---|
| Accuracy | 82% |
| ROC-AUC Score | 0.86 |

---

# Project Structure

```bash
customer-intelligence-system/
│
├── dashboard/
│   └── app.py
│
├── models/
│   ├── churn_pipeline.pkl
│   ├── logistic_regression_model.pkl
│   └── model_columns.pkl
│
├── notebooks/
├── src/
├── data/
│
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

---

# Installation & Setup

## Clone Repository

```bash
git clone https://github.com/SmrSingh/customer-intelligence-system.git
cd customer-intelligence-system
```

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Streamlit App Locally

```bash
streamlit run dashboard/app.py
```

---

# Docker Setup

## Build Docker Image

```bash
docker build -t churn-app .
```

## Run Docker Container

```bash
docker run -p 8501:8501 churn-app
```

---

# Deployment

The application is deployed using:
- Docker
- GitHub
- Render Cloud Platform

---

# Future Improvements

- Add multiple ML models for comparison
- Integrate XGBoost and Random Forest
- Add authentication system
- Build REST API endpoints
- Add customer segmentation analytics
- Implement automated retraining pipeline

---



