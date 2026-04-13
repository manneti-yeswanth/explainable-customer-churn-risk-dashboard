import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Churn Dashboard", layout="wide")

st.markdown(
    "<h2 style='color:#1f4e79;'>Customer Churn Risk Dashboard</h2>",
    unsafe_allow_html=True
)
st.caption("Predict churn risk and support business decisions")

# ---------------- LOAD ----------------
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

explainer = shap.TreeExplainer(model)

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
charge_ratio = monthly / (total + 100)   # stabilized
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

# ---------------- RISK FUNCTION ----------------
def get_risk(p):
    if p < 0.25:
        return "Low"
    elif p < 0.60:
        return "Medium"
    else:
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
        st.error("High churn risk detected. Immediate retention action required.")
    elif risk == "Medium":
        st.warning("Customer shows early churn signals. Proactive engagement recommended.")
    else:
        st.success("Customer is stable. No immediate action required.")

    # ================= RECOMMENDATION =================
    st.markdown("### 📌 Recommended Action")

    if risk == "High":
        st.error("Contact customer immediately, offer retention benefits, and provide dedicated support.")

    elif risk == "Medium":
        st.warning("Engage customer with better service, offers, or support to reduce churn risk.")

    else:
        st.success("Maintain regular engagement and provide loyalty benefits.")

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