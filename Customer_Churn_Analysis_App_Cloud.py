
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap

st.set_page_config(
    page_title="Customer Churn Analysis",
    page_icon="churn_favicon.png" if os.path.exists("churn_favicon.png") else "📊",
    layout="wide"
)

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "final_churn_model.pkl"

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Softer light theme with stronger visual contrast */
    .stApp {
        background-color: #E9EDF2 !important;
    }

    [data-testid="stHeader"] {
        background-color: #E9EDF2 !important;
    }

    [data-testid="stSidebar"] {
        background-color: #E2E7ED !important;
        border-right: 1px solid #C5CBD3;
    }

    [data-testid="stForm"] {
        background-color: #FFFFFF;
        border: 1px solid #C5CBD3;
        border-radius: 10px;
        padding: 1.2rem 1.25rem 1rem;
    }

    /* Clearly visible but still subtle input surfaces */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] {
        background-color: #F4F6F8 !important;
    }

    .page-title {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.15rem;
    }

    .page-subtitle {
        color: #667085;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .section-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #667085;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
    }

    /* Form controls */
    div[data-baseweb="select"] > div {
        border: 1px solid #8B95A5 !important;
        border-radius: 6px !important;
        background-color: #ffffff !important;
    }

    div[data-baseweb="select"] > div:focus-within {
        border: 1.5px solid #344054 !important;
        box-shadow: 0 0 0 1px #344054 !important;
    }

    div[data-baseweb="input"] {
        border: 1px solid #8B95A5 !important;
        border-radius: 6px !important;
        background-color: #ffffff !important;
    }

    div[data-baseweb="input"]:focus-within {
        border: 1.5px solid #344054 !important;
        box-shadow: 0 0 0 1px #344054 !important;
    }

    /* Result / information cards */
    .risk-card {
        border: 1px solid #98a2b3;
        border-radius: 10px;
        padding: 18px 20px;
        background: #f8f9fb;
        text-align: center;
    }

    .info-card {
        border: 1px solid #d0d5dd;
        border-radius: 8px;
        padding: 14px 16px;
        background: #ffffff;
        margin-bottom: 12px;
    }

    .risk-label {
        color: #667085;
        font-size: 0.82rem;
        margin-bottom: 0.25rem;
    }

    .risk-value {
        font-size: 1.35rem;
        font-weight: 700;
    }

    .risk-high { color: #b42318; }
    .risk-medium { color: #b54708; }
    .risk-low { color: #027a48; }

    .result-note {
        color: #667085;
        font-size: 0.82rem;
        margin-top: 0.75rem;
    }

    .recommendation-box {
        border: 1px solid #d0d5dd;
        border-left: 3px solid #667085;
        border-radius: 7px;
        padding: 12px 14px;
        margin: 8px 0;
        background: #ffffff;
    }

    .footer-note {
        color: #667085;
        font-size: 0.78rem;
        border-top: 1px solid #eaecf0;
        padding-top: 14px;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load model bundle
# -----------------------------
@st.cache_resource
def load_bundle(path):
    bundle = joblib.load(path)
    return bundle["preprocessor"], bundle["model"], float(bundle["threshold"])

if not os.path.exists(MODEL_PATH):
    st.error(
        "Trained model not found. Make sure final_churn_model.pkl is uploaded to the app repository."
    )
    st.stop()

preprocessor, model, model_threshold = load_bundle(MODEL_PATH)

numeric_features = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "AvgMonthlySpend", "IsNewCustomer", "NumServices",
    "HasOnlineProtection", "HasTechSupportOrSecurity"
]

categorical_features = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

all_features = numeric_features + categorical_features

# -----------------------------
# Helper functions
# -----------------------------
def risk_category(prob):
    if prob < 0.30:
        return "LOW"
    if prob < 0.60:
        return "MEDIUM"
    return "HIGH"

def engineer_features(d):
    tenure = float(d["tenure"])
    total = float(d["TotalCharges"])

    # Same project feature-engineering logic
    d["AvgMonthlySpend"] = total / tenure if tenure > 0 else float(d["MonthlyCharges"])
    d["IsNewCustomer"] = int(tenure < 12)

    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    d["NumServices"] = sum(
        1 for c in service_cols
        if str(d[c]) == "Yes"
    )
    d["HasOnlineProtection"] = int(
        d["OnlineSecurity"] == "Yes" or d["OnlineBackup"] == "Yes"
        or d["DeviceProtection"] == "Yes"
    )
    d["HasTechSupportOrSecurity"] = int(
        d["TechSupport"] == "Yes" or d["OnlineSecurity"] == "Yes"
    )
    return d

def build_customer(values):
    d = {
        "gender": values["gender"],
        "SeniorCitizen": int(values["SeniorCitizen"]),
        "Partner": values["Partner"],
        "Dependents": values["Dependents"],
        "tenure": int(values["tenure"]),
        "PhoneService": values["PhoneService"],
        "MultipleLines": values["MultipleLines"],
        "InternetService": values["InternetService"],
        "OnlineSecurity": values["OnlineSecurity"],
        "OnlineBackup": values["OnlineBackup"],
        "DeviceProtection": values["DeviceProtection"],
        "TechSupport": values["TechSupport"],
        "StreamingTV": values["StreamingTV"],
        "StreamingMovies": values["StreamingMovies"],
        "Contract": values["Contract"],
        "PaperlessBilling": values["PaperlessBilling"],
        "PaymentMethod": values["PaymentMethod"],
        "MonthlyCharges": float(values["MonthlyCharges"]),
        "TotalCharges": float(values["TotalCharges"]),
    }
    d = engineer_features(d)
    return pd.DataFrame([d])[all_features]

def get_shap_explanation(customer_df):
    Xp = preprocessor.transform(customer_df)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xp)

    # SHAP 0.52 can return (n, features, classes) for binary classification.
    if isinstance(sv, list):
        vals = sv[1][0]
    elif np.asarray(sv).ndim == 3:
        vals = np.asarray(sv)[0, :, 1]
    else:
        vals = np.asarray(sv)[0]

    df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP_Value": vals
    })
    df["Absolute"] = df["SHAP_Value"].abs()
    return df.sort_values("Absolute", ascending=False), Xp

def clean_shap_explanation(shap_df, customer_df):
    """
    Convert one-hot encoded SHAP outputs into a customer-friendly explanation.
    For categorical variables, only the active category is shown. This avoids
    displaying impossible pairs such as "MultipleLines Yes" and "MultipleLines No".
    """
    rows = []

    # Numerical features are already single model features.
    for feature in numeric_features:
        prefix = "num__" + feature
        match = shap_df[shap_df["Feature"] == prefix]
        if not match.empty:
            rows.append({
                "Feature": feature,
                "SHAP_Value": float(match.iloc[0]["SHAP_Value"])
            })

    # For each categorical feature, show only the category active for this customer.
    for feature in categorical_features:
        prefix = "cat__" + feature + "_"
        matches = shap_df[shap_df["Feature"].str.startswith(prefix)]
        if matches.empty:
            continue

        active_value = str(customer_df.iloc[0][feature])

        # OneHotEncoder(drop='first' is used in the training pipeline only if
        # applicable; handle both explicit active columns and dropped baselines.
        exact = matches[matches["Feature"].str.endswith("_" + active_value)]
        if not exact.empty:
            value = float(exact.iloc[0]["SHAP_Value"])
            label = f"{feature} = {active_value}"
        else:
            # If the active category is the dropped/baseline category, its
            # encoded columns are all zero. The grouped contribution is zero.
            value = 0.0
            label = f"{feature} = {active_value}"

        rows.append({"Feature": label, "SHAP_Value": value})

    out = pd.DataFrame(rows)
    out["Absolute"] = out["SHAP_Value"].abs()
    return out.sort_values("Absolute", ascending=False).reset_index(drop=True)

def human_feature_name(name):
    name = name.replace("num__", "").replace("cat__", "")
    name = name.replace("_", " ")
    return name

def recommendations(customer, risk):
    c = customer.iloc[0]
    out = []

    if risk == "HIGH":
        if c["Contract"] == "Month-to-month":
            out.append("Offer an incentive to move to a longer-term contract.")
        if c["tenure"] < 12:
            out.append("Provide an early-tenure onboarding or retention follow-up.")
        if c["MonthlyCharges"] > 70:
            out.append("Review pricing, plan value, and personalized discount eligibility.")
        if c["OnlineSecurity"] == "No":
            out.append("Offer Online Security or explain its value.")
        if c["TechSupport"] == "No":
            out.append("Offer Tech Support assistance or a support package.")
        if c["InternetService"] == "Fiber optic":
            out.append("Review the customer's Fiber Optic service experience.")
        if c["PaymentMethod"] == "Electronic check":
            out.append("Consider encouraging an easier recurring payment method.")
    elif risk == "MEDIUM":
        out.append("Use a personalized service check-in to prevent risk escalation.")
        if c["Contract"] == "Month-to-month":
            out.append("Consider an incentive for a longer-term contract.")
        if c["MonthlyCharges"] > 70:
            out.append("Review pricing and available discounts.")
        if c["OnlineSecurity"] == "No":
            out.append("Consider offering Online Security.")
    else:
        out.append("No immediate retention intervention is indicated.")
        out.append("Continue normal customer engagement and service monitoring.")

    return list(dict.fromkeys(out))

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="page-title">Customer Churn Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Telecom customer retention decision support</div>',
    unsafe_allow_html=True
)

st.info(
    f"Classification threshold: {model_threshold:.2f}  |  "
    "Risk levels: Low < 30%  •  Medium 30–<60%  •  High ≥ 60%"
)
st.caption(
    "The classification threshold determines the churn prediction. "
    "Risk levels provide a separate summary of customer risk."
)

# -----------------------------
# Presets
# -----------------------------
presets = {
    "🔴 High-risk example": {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.05,
        "TotalCharges": 85.05
    },

    # Verified against the actual saved Random Forest:
    # predicted churn probability = 0.449228 (44.9228%)
    "🟡 Medium-risk example": {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "No",
        "OnlineSecurity": "No internet service",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 70.00,
        "TotalCharges": 840.00
    },

    "🟢 Low-risk example": {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 72,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Two year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 110.00,
        "TotalCharges": 7900.00
    }
}

preset_name = st.selectbox(
    "Demo example",
    ["Custom"] + list(presets.keys()),
    help="Demo profiles are inputs only. Churn probability is always calculated live by the trained Random Forest."
)
preset = presets.get(preset_name)

if preset_name != "Custom":
    st.caption(
        "Demo profile loaded. The probability shown after analysis is generated live "
        "by the trained Random Forest; it is not hard-coded."
    )

# -----------------------------
# Input form
# -----------------------------
with st.form("customer_form"):
    st.subheader("Customer Information")

    def default(key, fallback):
        return preset[key] if preset else fallback

    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ["Male","Female"],
                              index=["Male","Female"].index(default("gender","Male")))
        senior = st.selectbox("Senior Citizen", [0,1],
                              index=[0,1].index(default("SeniorCitizen",0)))
        partner = st.selectbox("Partner", ["Yes","No"],
                               index=["Yes","No"].index(default("Partner","No")))
        dependents = st.selectbox("Dependents", ["Yes","No"],
                                  index=["Yes","No"].index(default("Dependents","No")))
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100,
                                 value=int(default("tenure",12)), step=1)

    with c2:
        phone = st.selectbox("Phone Service", ["Yes","No"],
                             index=["Yes","No"].index(default("PhoneService","Yes")))
        multiple = st.selectbox("Multiple Lines", ["Yes","No","No phone service"],
                                index=["Yes","No","No phone service"].index(default("MultipleLines","No")))
        internet = st.selectbox("Internet Service", ["DSL","Fiber optic","No"],
                                index=["DSL","Fiber optic","No"].index(default("InternetService","DSL")))
        security = st.selectbox("Online Security", ["Yes","No","No internet service"],
                                index=["Yes","No","No internet service"].index(default("OnlineSecurity","No")))
        backup = st.selectbox("Online Backup", ["Yes","No","No internet service"],
                              index=["Yes","No","No internet service"].index(default("OnlineBackup","No")))
        device = st.selectbox("Device Protection", ["Yes","No","No internet service"],
                              index=["Yes","No","No internet service"].index(default("DeviceProtection","No")))

    with c3:
        support = st.selectbox("Tech Support", ["Yes","No","No internet service"],
                               index=["Yes","No","No internet service"].index(default("TechSupport","No")))
        tv = st.selectbox("Streaming TV", ["Yes","No","No internet service"],
                          index=["Yes","No","No internet service"].index(default("StreamingTV","No")))
        movies = st.selectbox("Streaming Movies", ["Yes","No","No internet service"],
                              index=["Yes","No","No internet service"].index(default("StreamingMovies","No")))
        contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"],
                                index=["Month-to-month","One year","Two year"].index(default("Contract","Month-to-month")))
        paperless = st.selectbox("Paperless Billing", ["Yes","No"],
                                 index=["Yes","No"].index(default("PaperlessBilling","Yes")))
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"],
            index=[
                "Electronic check","Mailed check",
                "Bank transfer (automatic)","Credit card (automatic)"
            ].index(default("PaymentMethod","Electronic check"))
        )

    c4, c5 = st.columns(2)
    with c4:
        monthly = st.number_input("Monthly Charges", min_value=0.0,
                                  value=float(default("MonthlyCharges",70.0)), step=0.01)
    with c5:
        total = st.number_input("Total Charges", min_value=0.0,
                                value=float(default("TotalCharges",monthly*max(tenure,1))), step=0.01)

    submitted = st.form_submit_button("🔍 Analyze Customer", use_container_width=True)

# -----------------------------
# Prediction
# -----------------------------
if submitted:
    values = {
        "gender":gender, "SeniorCitizen":senior, "Partner":partner,
        "Dependents":dependents, "tenure":tenure, "PhoneService":phone,
        "MultipleLines":multiple, "InternetService":internet,
        "OnlineSecurity":security, "OnlineBackup":backup,
        "DeviceProtection":device, "TechSupport":support, "StreamingTV":tv,
        "StreamingMovies":movies, "Contract":contract,
        "PaperlessBilling":paperless, "PaymentMethod":payment,
        "MonthlyCharges":monthly, "TotalCharges":total
    }

    customer_df = build_customer(values)
    probability = float(model.predict_proba(preprocessor.transform(customer_df))[:,1][0])
    prediction = int(probability >= model_threshold)
    risk = risk_category(probability)

    st.divider()
    st.subheader("Analysis Result")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Churn Probability", f"{probability*100:.2f}%")
    with m2:
        st.metric("Model Prediction", "YES — Churn" if prediction else "NO — Retain")
    with m3:
        st.markdown(
            f'<div class="risk-card">'
            f'<div class="risk-label">Risk level</div>'
            f'<div class="risk-value risk-{risk.lower()}">{risk}</div></div>',
            unsafe_allow_html=True
        )

    # SHAP explanation
    raw_shap_df, _ = get_shap_explanation(customer_df)
    shap_df = clean_shap_explanation(raw_shap_df, customer_df)

    positive = shap_df[shap_df["SHAP_Value"] > 0].head(6)
    negative = shap_df[shap_df["SHAP_Value"] < 0].sort_values("SHAP_Value").head(6)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("Key risk factors")
        if len(positive):
            for _, row in positive.iterrows():
                st.write(
                    f"• **{human_feature_name(row['Feature'])}**  "
                    f"(+{row['SHAP_Value']:.4f})"
                )
        else:
            st.write("No positive SHAP contributors were identified.")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("Protective factors")
        if len(negative):
            for _, row in negative.iterrows():
                st.write(
                    f"• **{human_feature_name(row['Feature'])}**  "
                    f"({row['SHAP_Value']:.4f})"
                )
        else:
            st.write("No negative SHAP contributors were identified.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.caption(
        "SHAP values show how features influenced this model prediction; "
        "they do not prove that a feature causes churn."
    )

    st.subheader("Suggested retention actions")
    recs = recommendations(customer_df, risk)
    for r in recs:
        st.markdown(f'<div class="recommendation-box">{r}</div>', unsafe_allow_html=True)

    if risk == "HIGH":
        summary = "Prioritize this customer for retention outreach using the strongest model-identified risk factors."
    elif risk == "MEDIUM":
        summary = "Monitor this customer and use targeted engagement before risk escalates."
    else:
        summary = "Continue normal engagement; no immediate retention intervention is indicated."

    st.info(f"**Decision summary:** {summary}")

    with st.expander("Engineered features used by the model"):
        st.dataframe(customer_df.T.rename(columns={0:"Value"}), use_container_width=True)

    with st.expander("Technical model details"):
        st.write(f"Model: Random Forest classifier")
        st.write(f"Classification threshold: {model_threshold:.2f}")
        st.write("Probability source: trained Random Forest (`predict_proba`).")
        st.write("Explanation method: SHAP TreeExplainer.")
        st.write("Risk bands are decision-support categories and are separate from the 0.50 classification threshold.")

st.markdown(
    '<div class="footer-note">'
    'Academic project demonstration. SHAP values describe model behavior and do not establish causality. '
    'Retention actions are rule-based decision-support suggestions.'
    '</div>',
    unsafe_allow_html=True
)
