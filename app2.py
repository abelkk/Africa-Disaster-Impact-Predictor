
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load("noteboooks\xgb.joblib")

model = load_model()

# Page configuration
st.set_page_config(
    page_title="Africa Disaster Impact Predictor",
    page_icon="",
    layout="centered"
)

# App title
st.title(" Africa Disaster Impact Predictor")

st.write(
    "Enter disaster characteristics to estimate "
    "the potential number of deaths."
)

# User inputs — replace these with your actual features
disaster_type = st.selectbox(
    "Disaster Type",
    ["Flood", "Storm", "Drought", "Epidemic", "Other"]
)

start_year = st.number_input(
    "Start Year",
    min_value=1900,
    max_value=2100,
    value=2025
)

start_month = st.selectbox(
    "Start Month",
    list(range(1, 13))
)

# Prediction button
if st.button("Predict Estimated Deaths"):

    # Example only: use the exact training features
    # and their original order in your actual app.
    input_data = pd.DataFrame([{
        "Disaster Type": disaster_type,
        "Start Year": start_year,
        "Start Month": start_month
    }])

    try:
        prediction = model.predict(input_data)

        estimated_deaths = np.expm1(prediction[0])

        # Death counts cannot be negative
        estimated_deaths = max(0, estimated_deaths)

        st.success(
            f"Predicted deaths: {estimated_deaths:,.0f}"
        )

        st.caption(
            "This is a model estimate, not a guaranteed outcome."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Check that your inputs match the model's "
            "training features and preprocessing."
        )
