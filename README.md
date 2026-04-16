# 📊 Customer Churn Risk Intelligence Dashboard

A complete end-to-end machine learning system to **predict customer churn risk and explain the reasons behind it**, enabling businesses to take proactive retention actions.

🔗 **Live Demo:** https://explainable-customer-churn-risk-dashboard-h8unhhkjcvpyjoib73me.streamlit.app/

🚀 Built with XGBoost + SHAP + Streamlit for real-time prediction and explainable insights

---

## 🎯 Problem Statement

Customer churn is one of the biggest challenges for subscription-based businesses.

Most systems only predict churn —
👉 but fail to explain *why it happens*

This project addresses both:

* ✔ Predict churn risk
* ✔ Explain key drivers
* ✔ Enable actionable business decisions

---

## 📂 Dataset

The dataset contains customer demographic and behavioral data such as:

* Tenure
* Monthly charges
* Contract type
* Internet service
* Payment method

These features help identify patterns associated with customer churn.

---

## 🎯 Project Objective

* Identify customers likely to churn
* Provide interpretable insights using SHAP
* Enable data-driven retention strategies

---

## ⚙️ How It Works

1. User inputs customer details via Streamlit UI
2. Data is preprocessed and aligned with trained features
3. XGBoost model predicts churn probability
4. SHAP explains feature contributions
5. System assigns risk level and suggests business action

---

## 🧠 Solution Overview

This system combines:

* **XGBoost Model** → High-performance prediction
* **Feature Engineering** → Behavior-based insights
* **SHAP Explainability** → Transparent predictions
* **Streamlit Dashboard** → Interactive UI

---

## 📊 Model Performance

| Metric    | Value     |
| --------- | --------- |
| Accuracy  | **76.7%** |
| ROC-AUC   | **0.81**  |
| Precision | **0.56**  |
| Recall    | **0.61**  |
| F1 Score  | **0.58**  |

📌 **Key Insight:**

* Strong overall performance
* Moderate recall for churn class → indicates scope for improvement
* Model slightly biased toward non-churn customers

---

## 📸 Application Walkthrough

### 🔹 User Input Interface

![Input UI](./input_ui.png)

---

### 🔹 Risk Prediction Output

![Risk Output](./risk_output.png)

* Risk Level (Low / Medium / High)
* Probability Score
* Action Recommendation

---

### 🔹 Feature Importance

![Feature Importance](./feature_importance.png)

Top drivers:

* Tenure
* Contract type
* Payment method

---

### 🔹 SHAP Explainability

![SHAP](./shap_explanation.png)

Shows how each feature contributes to churn prediction.

---

### 🔹 Key Drivers Summary

![Key Drivers](./key_drivers.png)

Highlights most impactful churn factors.

---

## 💡 Key Insights from Model

* New customers are more likely to churn
* High monthly charges increase churn risk
* Contract type significantly impacts retention
* Customer engagement strongly affects churn

---

## 📌 Business Recommendations

### 🔴 High Risk Customers

* Immediate outreach
* Personalized retention offers

### 🟡 Medium Risk Customers

* Improve engagement
* Offer incentives

### 🟢 Low Risk Customers

* Maintain satisfaction
* Loyalty programs

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* XGBoost
* SHAP
* Matplotlib
* Streamlit

---

## ▶️ Run Locally

```bash
streamlit run app.py
```

---

## 🔥 What Makes This Project Strong

This is not just a prediction model —
it is a **complete decision-support system**:

* ✔ Prediction
* ✔ Explainability
* ✔ Business action

---

## 📈 Business Impact

* Enables early churn detection
* Supports targeted retention strategies
* Reduces revenue loss

---

## 🔮 Future Improvements

* Improve recall using advanced resampling
* Hyperparameter tuning
* Cloud deployment (AWS / GCP)
* API integration

---

## 👨‍💻 Author

**Yeshwanth Reddy M**
🔗 GitHub: https://github.com/manneti-yeswanth
🔗 LinkedIn: www.linkedin.com/in/manneti-yeswanth-reddy-2758693b6
