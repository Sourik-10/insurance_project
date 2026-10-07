import os

import requests
import streamlit as st


DEFAULT_API_URL = os.getenv("INSURANCE_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Insurance Charge Predictor", page_icon="🩺", layout="centered")
st.markdown("""
<style>
    .block-container {max-width: 900px; padding-top: 2.2rem; padding-bottom: 3rem;}
    .hero {padding: 1.5rem 1.7rem; border-radius: 18px; margin-bottom: 1.4rem;
        background: linear-gradient(125deg, #102a43 0%, #176b74 100%); color: white;}
    .hero h1 {color: white; font-size: 2rem; margin: 0 0 .45rem 0;}
    .hero p {color: #e2f3f2; margin: 0; font-size: 1rem;}
    .hint {color: #627487; font-size: .88rem; padding-top: .15rem;}
    div[data-testid="stForm"] {border: 1px solid #dce5eb; border-radius: 16px; padding: 1.2rem 1.4rem;}
    div[data-testid="stMetric"] {background: #f4f8fa; border: 1px solid #e2eaef; padding: 1rem; border-radius: 12px;}
    div.stButton > button {border-radius: 10px; font-weight: 650; min-height: 3rem;}
    [data-testid="stSidebar"] {border-right: 1px solid #e4ebef;}
</style>
<div class="hero">
  <h1>🩺 Insurance Charge Predictor</h1>
  <p>Enter a few details to receive an estimated annual insurance charge.</p>
</div>
""", unsafe_allow_html=True)
st.caption("BMI is calculated from height and weight. This model estimate is not an insurance quote.")

with st.sidebar:
    st.subheader("API connection")
    api_url = st.text_input("FastAPI base URL", value=DEFAULT_API_URL).strip().rstrip("/")
    st.caption("Start the API with `uvicorn api:app --reload`.")

with st.form("prediction_form", border=True):
    st.subheader("Your details")
    st.markdown('<p class="hint">All fields are used to estimate your annual charges.</p>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Personal information**")
        age = st.number_input("Age (years)", min_value=18, max_value=64, value=30, step=1)
        sex = st.selectbox("Sex", ["female", "male"])
        children = st.number_input("Covered children", min_value=0, max_value=10, value=0, step=1)
    with right:
        st.markdown("**Health and location**")
        weight_kg = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=75.0, step=0.5)
        height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=230.0, value=170.0, step=0.5)
        smoker = st.selectbox("Smoking status", ["no", "yes"])
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
    st.caption("The model supports ages 18–64. Height and weight are used to calculate BMI.")
    submitted = st.form_submit_button("Calculate estimate", type="primary", use_container_width=True)

if submitted:
    bmi = float(weight_kg) / (float(height_cm) / 100) ** 2
    if not 10 < bmi < 60:
        st.error(f"Calculated BMI ({bmi:.1f}) is outside the model's supported range (10–60). Check the height and weight values.")
        st.stop()

    payload = {
        "age": int(age),
        "sex": sex,
        "bmi": float(bmi),
        "children": int(children),
        "smoker": smoker,
        "region": region,
    }
    try:
        response = requests.post(f"{api_url}/predict", json=payload, timeout=15)
        if response.ok:
            result = response.json()
            st.markdown("### Your estimate")
            st.success("Estimate calculated successfully")
            st.metric("Estimated annual charges", f"${result['predicted_charges']:,.2f}")
            col1, col2 = st.columns(2)
            col1.metric("Calculated BMI", f"{bmi:.1f}", result["bmi_category"].title())
            col2.metric("Smoker and obese indicator", "Yes" if result["smoker_obese_risk"] else "No")
        else:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            st.error(f"API returned {response.status_code}: {detail}")
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to {api_url}. Make sure the FastAPI server is running.")
    except requests.exceptions.Timeout:
        st.error("The API took too long to respond. Please try again.")
    except requests.exceptions.RequestException as exc:
        st.error(f"Request failed: {exc}")
