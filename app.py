import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Africa Disaster Impact Predictor", page_icon="", layout="wide")

st.title(" Africa Disaster Impact Predictor")
st.write(
    "Objective: Explore historical disaster records and estimate deaths using a machine-learning model "
    
)

st.sidebar.header("1. Load the  dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload the EM-DAT Excel file (.xlsx)", type=["xlsx"]
)


@st.cache_data(show_spinner=False)
def read_data(file_bytes):
    import io
    return pd.read_excel(io.BytesIO(file_bytes))

@st.cache_resource(show_spinner=False)
def train_model(data, model_choice):
    df = data.copy()

    required = ["Estimated Deaths", "Disaster Type", "Country"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError("Missing required columns: " + ", ".join(missing_required))

    
    candidate_features = [
        "Historic", "Disaster Type", "Country", "Subregion", "Appeal",
        "Declaration", "Start Year", "Start Month", "Start Day", "OFDA/BHA Response"
    ]
    features = [c for c in candidate_features if c in df.columns]
    data_model = df[features + ["Estimated Deaths"]].copy()

    data_model["Estimated Deaths"] = pd.to_numeric(
        data_model["Estimated Deaths"], errors="coerce"
    )
    data_model = data_model.dropna(subset=["Estimated Deaths"])
    data_model = data_model[data_model["Estimated Deaths"] >= 0]

    if len(data_model) < 20:
        raise ValueError("At least 20 rows with valid Estimated Deaths are needed to train the model.")

    
    for col in ["Historic", "Appeal", "Declaration", "OFDA/BHA Response"]:
        if col in data_model.columns:
            data_model[col] = data_model[col].replace({
                "Yes": "Yes", "No": "No", 1: "Yes", 0: "No",
                True: "Yes", False: "No"
            })

    X = data_model[features]
    y = np.log1p(data_model["Estimated Deaths"])

    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_features = [c for c in X.columns if c not in categorical_features]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    if model_choice == "XGBoost":
        try:
            from xgboost import XGBRegressor
            estimator = XGBRegressor(
                n_estimators=250, max_depth=5, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror",
                random_state=42, n_jobs=-1
            )
        except ImportError:
            raise ValueError("XGBoost is not installed. Install it with: pip install xgboost")
    else:
        estimator = RandomForestRegressor(
            n_estimators=250, max_depth=15, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )

    model = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    pred_log = model.predict(X_test)
    actual = np.expm1(y_test)
    predicted = np.maximum(0, np.expm1(pred_log))
    metrics = {
        "R² (log scale)": r2_score(y_test, pred_log),
        "MAE (deaths)": mean_absolute_error(actual, predicted),
        "RMSE (deaths)": np.sqrt(mean_squared_error(actual, predicted))
    }
    return model, features, data_model, metrics

if uploaded_file is None:
    st.info("Upload your EM-DAT Excel dataset in the sidebar to train the model and for prediction .")
    st.subheader("What this app does")
    c1, c2, c3 = st.columns(3)
    c1.metric("Input", "Historical data")
    c2.metric("Model", "Random Forest / XGBoost")
    c3.metric("Output", "Estimated deaths")
    st.markdown(
        "**Expected target:** `Estimated Deaths`  \n"
        "**Predictors:** disaster type, country, subregion (if available), historical status, "
        "appeal/declaration status, start date, and OFDA/BHA response status."
    )
    st.stop()

try:
    raw_df = read_data(uploaded_file.getvalue())
    st.success(f"Dataset loaded: {raw_df.shape[0]:,} rows and {raw_df.shape[1]} columns.")
    with st.expander("Preview dataset"):
        st.dataframe(raw_df.head(10), use_container_width=True)

    model_choice = st.sidebar.selectbox("Choose model", ["Random Forest", "XGBoost"])
    with st.spinner("Training and evaluating the model on an 80/20 split..."):
        model, feature_cols, model_data, metrics = train_model(raw_df, model_choice)

    st.subheader("Model evaluation")
    m1, m2, m3 = st.columns(3)
    m1.metric("R² (log scale)", f"{metrics['R² (log scale)']:.3f}")
    m2.metric("MAE", f"{metrics['MAE (deaths)']:,.1f} deaths")
    m3.metric("RMSE", f"{metrics['RMSE (deaths)']:,.1f} deaths")
    st.caption(
        "Metrics are calculated on a held-out test split. Death counts can be highly skewed, "
        "so inspect errors on severe events separately before drawing conclusions."
    )

    st.divider()
    st.subheader("2. Estimate deaths for a disaster scenario")
    st.write("Choose the scenario details below. Categories not present during training are handled by the model, but predictions for unfamiliar cases may be less reliable.")

    source = model_data
    def categories(col, default="Unknown"):
        if col not in source.columns:
            return [default]
        vals = source[col].dropna().astype(str).unique().tolist()
        vals = sorted(vals)
        return vals if vals else [default]

    country_options = categories("Country")
    disaster_options = categories("Disaster Type")
    subregion_options = categories("Subregion")
    yes_no_options = ["No", "Yes"]

    with st.form("prediction_form"):
        left, middle, right = st.columns(3)
        with left:
            disaster_type = st.selectbox("Disaster type", disaster_options)
            country = st.selectbox("Country", country_options)
            subregion = st.selectbox("Subregion", subregion_options) if "Subregion" in feature_cols else None
            start_year = st.number_input("Start year", min_value=1900, max_value=2100, value=2026) if "Start Year" in feature_cols else None
        with middle:
            start_month = st.selectbox("Start month", list(range(1, 13)), index=0) if "Start Month" in feature_cols else None
            start_day = st.selectbox("Start day", list(range(1, 32)), index=0) if "Start Day" in feature_cols else None
            historic = st.selectbox("Historic event?", yes_no_options) if "Historic" in feature_cols else None
            appeal = st.selectbox("Humanitarian appeal?", yes_no_options) if "Appeal" in feature_cols else None
        with right:
            declaration = st.selectbox("Emergency declaration?", yes_no_options) if "Declaration" in feature_cols else None
            response = st.selectbox("OFDA/BHA response?", yes_no_options) if "OFDA/BHA Response" in feature_cols else None

        submitted = st.form_submit_button("Estimate deaths", type="primary")

    if submitted:
        row = {}
        for col in feature_cols:
            mapping = {
                "Disaster Type": disaster_type,
                "Country": country,
                "Subregion": subregion,
                "Start Year": start_year,
                "Start Month": start_month,
                "Start Day": start_day,
                "Historic": historic,
                "Appeal": appeal,
                "Declaration": declaration,
                "OFDA/BHA Response": response
            }
            row[col] = mapping.get(col, np.nan)
        input_df = pd.DataFrame([row], columns=feature_cols)
        predicted_log = float(model.predict(input_df)[0])
        estimate = max(0, float(np.expm1(predicted_log)))
        st.success("Prediction complete")
        st.metric("Predicted estimated deaths", f"{estimate:,.0f}")
        st.warning(
            "This is a statistical estimate based on historical records, not a reliable standalone "
            "prediction of future fatalities. Actual outcomes depend on exposure, vulnerability, "
            "warning systems, response capacity, and many factors not represented here."
        )

    st.divider()
    st.subheader("3. Save the trained model")
    st.write("Save the preprocessing steps and model together so future predictions use the same feature encoding.")
    if st.button("Prepare model download"):
        model_bytes = __import__("io").BytesIO()
        joblib.dump({"pipeline": model, "feature_columns": feature_cols}, model_bytes)
        st.download_button(
            "Download trained model (.joblib)",
            data=model_bytes.getvalue(),
            file_name="africa_disaster_deaths_pipeline.joblib",
            mime="application/octet-stream"
        )

except Exception as exc:
    st.error(f"Could not prepare the app: {exc}")
    st.info(
        "Check that the uploaded workbook contains the expected EM-DAT columns, especially "
        "`Estimated Deaths`, `Disaster Type`, and `Country`."
    )
