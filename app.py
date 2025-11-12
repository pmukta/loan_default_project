import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Loan Default Predictor", page_icon="💰", layout="centered")

# --- FIXED HEADER (adjusted to show fully) ---
st.markdown("""
    <style>
    .sticky-header {
        position: fixed;
        top: 3.5rem; /* pushes header slightly below Streamlit toolbar */
        width: 100%;
        background-color: white;
        z-index: 999;
        padding: 0.6rem 0;
        border-bottom: 2px solid #e0e0e0;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 600;
    }
    .main-content {
        margin-top: 120px; /* leave enough space for fixed header */
    }
    </style>
""", unsafe_allow_html=True)

# --- Sticky Header HTML ---
st.markdown('<div class="sticky-header">💰 Loan Default Prediction App</div>', unsafe_allow_html=True)

# --- Main Content (below header) ---
st.markdown('<div class="main-content">', unsafe_allow_html=True)
st.markdown("### Predict whether a loan will be repaid or defaulted.")
st.markdown("Fill in the applicant details below 👇")
st.divider()


# ---- Input Form Layout ----
col1, col2 = st.columns(2)

with col1:
    credit_policy = st.selectbox("Credit Policy", [1, 0], format_func=lambda x: "✅ Meets Policy" if x == 1 else "❌ Does Not Meet")
    purpose = st.selectbox(
        "Loan Purpose",
        ["credit_card", "debt_consolidation", "educational", "home_improvement", "major_purchase", "small_business"]
    )
    int_rate = st.slider("Interest Rate", 0.0, 0.30, 0.12, 0.01)
    installment = st.number_input("Installment Amount", min_value=0.0, step=10.0, value=200.0)
    annual_inc = st.number_input("Annual Income (₹)", min_value=10000.0, step=5000.0, value=500000.0)
    dti = st.slider("Debt-to-Income Ratio (DTI)", 0.0, 50.0, 15.0, 0.1)

with col2:
    fico = st.slider("FICO Score", 300, 850, 700)
    days_with_cr_line = st.number_input("Days with Credit Line", min_value=0.0, step=10.0, value=3000.0)
    revol_bal = st.number_input("Revolving Balance", min_value=0.0, step=100.0, value=5000.0)
    revol_util = st.slider("Revolving Utilization (%)", 0.0, 150.0, 30.0, 1.0)
    inq_last_6mths = st.number_input("Inquiries in Last 6 Months", min_value=0, step=1, value=1)
    delinq_2yrs = st.number_input("Delinquencies in Last 2 Years", min_value=0, step=1, value=0)
    pub_rec = st.number_input("Public Records", min_value=0, step=1, value=0)

st.divider()

# ---- Prepare Input ----
log_annual_inc = np.log(annual_inc)

purpose_cols = [
    "purpose_credit_card",
    "purpose_debt_consolidation",
    "purpose_educational",
    "purpose_home_improvement",
    "purpose_major_purchase",
    "purpose_small_business"
]

purpose_dict = dict.fromkeys(purpose_cols, 0)
purpose_dict[f"purpose_{purpose}"] = 1

sample = pd.DataFrame([{
    'credit.policy': credit_policy,
    'int.rate': int_rate,
    'installment': installment,
    'log.annual.inc': log_annual_inc,
    'dti': dti,
    'fico': fico,
    'days.with.cr.line': days_with_cr_line,
    'revol.bal': revol_bal,
    'revol.util': revol_util,
    'inq.last.6mths': inq_last_6mths,
    'delinq.2yrs': delinq_2yrs,
    'pub.rec': pub_rec,
    **purpose_dict
}])

# ---- Prediction ----
model = pickle.load(open("loan_default_model.pkl", "rb"))

if st.button("🔍 Predict Loan Status", use_container_width=True):
    prediction = model.predict(sample)[0]
    confidence = model.predict_proba(sample).max() * 100

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ The loan is **LIKELY TO DEFAULT**.\nConfidence: **{confidence:.2f}%**")
        st.markdown("💡 *Consider revising credit policy or reducing interest burden.*")
    else:
        st.success(f"✅ The loan is **LIKELY TO BE REPAID**.\nConfidence: **{confidence:.2f}%**")
        st.markdown("🎉 *Good profile — fits within credit norms.*")

st.divider()
st.caption("Model trained using Random Forest Classifier with balanced class weights.")
st.markdown('</div>', unsafe_allow_html=True)
