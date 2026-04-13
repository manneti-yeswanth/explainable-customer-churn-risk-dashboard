import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# ---------------- LOAD ----------------
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# ✅ FIXED: correct SHAP explainer for XGBoost
explainer = shap.TreeExplainer(model)

# ---------------- TITLE ----------------
st.title("Customer Churn Risk Intelligence Dashboard")
st.write("Analyze customer churn risk and support retention decisions.")

# ---------------- INPUT ----------------
st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.number_input("Monthly Charges", 20.0, 200.0, 50.0)
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

with col2:
    total = st.number_input("Total Charges", 0.0, 10000.0, 500.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    tech = st.selectbox("Tech Support", ["Yes", "No"])

# ---------------- FEATURE ENGINEERING ----------------
charge_ratio = monthly / (total + 1)
activity_score = tenure * monthly

data = {
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "ChargeRatio": charge_ratio,
    "ActivityScore": activity_score,
    "Contract": contract,
    "InternetService": internet,
    "TechSupport": tech
}

df = pd.DataFrame([data])

# ---------------- ENCODING ----------------
df = pd.get_dummies(df)
df = df.reindex(columns=columns, fill_value=0)

# ---------------- HELPERS ----------------
def get_risk(p):
    if p < 0.3:
        return "Low"
    elif p < 0.7:
        return "Medium"
    return "High"

# ---------------- PREDICT ----------------
if st.button("Predict"):

    prob = model.predict_proba(df)[0][1]
    risk = get_risk(prob)

    st.markdown("---")

    # ================= RESULT =================
    st.markdown("### 📊 Risk Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Risk Level", risk)
    c2.metric("Probability", f"{prob:.2f}")
    c3.metric("Risk Score", f"{prob*100:.1f}%")

    # ================= DECISION =================
    st.markdown("### 🏦 Business Decision")

    if risk == "High":
        st.error("🔴 High churn risk detected. Immediate retention action required.")
    elif risk == "Medium":
        st.warning("🟡 Customer shows early churn signals. Proactive engagement recommended.")
    else:
        st.success("🟢 Customer is stable. No immediate action required.")

    # ================= ACTION =================
    st.markdown("### 📌 Recommended Action")

    if tenure < 12:
        st.info("Improve onboarding experience to reduce early churn.")
    elif monthly > 80:
        st.info("Customer may be price sensitive. Consider personalized discount.")
    elif contract == "Month-to-month":
        st.info("Encourage customer to move to long-term contract.")
    else:
        st.info("Maintain engagement through loyalty programs.")

    # ================= FEATURE IMPORTANCE =================
    st.markdown("### 📊 Feature Importance")

    importances = model.feature_importances_

    imp_df = pd.DataFrame({
        "Feature": columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(8)

    fig, ax = plt.subplots()
    ax.barh(imp_df["Feature"], imp_df["Importance"])
    ax.set_title("Top Factors Driving Churn")
    ax.invert_yaxis()

    st.pyplot(fig)

    # ================= SHAP =================
    st.markdown("### 🧠 Why this prediction")

    # ✅ FIXED: correct way
    shap_values = explainer(df)

    fig = plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

    # ================= KEY DRIVERS =================
    st.markdown("### 📌 Key Drivers")

    shap_df = pd.DataFrame({
        "Feature": df.columns,
        "Impact": shap_values.values[0]
    })

    shap_df["abs"] = shap_df["Impact"].abs()
    shap_df = shap_df.sort_values(by="abs", ascending=False).head(3)

    for _, row in shap_df.iterrows():
        name = row["Feature"].replace("_", " ")

        if row["Impact"] > 0:
            st.error(f"{name} is increasing churn risk")
        else:
            st.success(f"{name} is helping retain the customer")
