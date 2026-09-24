"""
app.py
------
Streamlit frontend for the Job Salary Prediction project.

How to run:
    1. Train the model first:
           cd salary_prediction
           python model_training.py
    2. Launch the app:
           streamlit run app.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ── Path setup ────────────────────────────────────────────────────────────────
APP_DIR  = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

from data_processing import (
    load_data, clean_data,
    load_encoders, encode_single_row,
    FEATURE_COLS,
)

ASSETS_DIR   = os.path.join(APP_DIR, "assets")
MODELS_DIR   = os.path.join(APP_DIR, "models")
DATASET_PATH = os.path.join(APP_DIR, "job_salary_prediction_dataset.csv")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Job Salary Prediction",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Main background */
    .main { background-color: #f7f8fa; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px 16px;
    }

    /* Prediction result box */
    .prediction-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #3b82d4 100%);
        border-radius: 12px;
        padding: 28px;
        text-align: center;
        color: white;
        margin-top: 12px;
    }
    .prediction-box h1 { color: white; font-size: 2.6rem; margin: 0; }
    .prediction-box p  { color: #d0e8ff; font-size: 1rem; margin: 4px 0 0 0; }

    /* Section headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1f2328;
        border-left: 4px solid #3b82d4;
        padding-left: 10px;
        margin-bottom: 14px;
    }

    /* Info banner */
    .info-banner {
        background: #eef5ff;
        border: 1px solid #bfd7f5;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.9rem;
        color: #1f2328;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Cached helpers ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def load_model():
    path = os.path.join(MODELS_DIR, "best_model.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_resource(show_spinner="Loading encoders …")
def get_encoders():
    path = os.path.join(MODELS_DIR, "encoders.pkl")
    if not os.path.exists(path):
        return None
    return load_encoders(path)


@st.cache_data(show_spinner="Loading dataset …")
def get_clean_data():
    df = load_data(DATASET_PATH)
    return clean_data(df)


@st.cache_data
def load_results():
    path = os.path.join(MODELS_DIR, "results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_best_model_name():
    path = os.path.join(MODELS_DIR, "best_model_name.txt")
    if not os.path.exists(path):
        return "Unknown"
    with open(path) as f:
        return f.read().strip()


def asset(name: str) -> str:
    return os.path.join(ASSETS_DIR, name)


# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/fluency/96/000000/salary.png",
    width=70,
)
st.sidebar.title("💼 Job Salary Predictor")
st.sidebar.markdown("---")

pages = {
    "🏠 Home & Predict":     "predict",
    "📊 Dataset Overview":   "overview",
    "📈 EDA Visualizations": "eda",
    "🤖 Model Performance":  "model",
}
page = st.sidebar.radio("Navigate", list(pages.keys()), label_visibility="collapsed")
active = pages[page]

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='font-size:0.78rem;color:#57606a;'>
    <b>How to use</b><br>
    1. Run <code>python model_training.py</code><br>
    2. Come back here<br>
    3. Fill in the form → Predict!
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Load artefacts ────────────────────────────────────────────────────────────
model    = load_model()
encoders = get_encoders()
df       = get_clean_data()
results  = load_results()
best_name = load_best_model_name()

model_ready = (model is not None) and (encoders is not None)

if not model_ready:
    st.warning(
        "⚠️  Trained model not found.  "
        "Please run `python model_training.py` inside the `salary_prediction/` folder first.",
        icon="⚠️",
    )


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 1 – Home & Predict
# ════════════════════════════════════════════════════════════════════════════════
if active == "predict":
    # ── Hero header ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style='background:#1e3a5f;border-radius:12px;padding:28px 32px;margin-bottom:24px;'>
            <h1 style='color:#ffffff;margin:0;font-size:2rem;'>💼 Job Salary Prediction</h1>
            <p style='color:#bfd7f5;margin:6px 0 0 0;font-size:1rem;'>
                Enter employee / job details below to get an instant salary estimate
                powered by a Machine Learning regression model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Input form ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Employee / Job Information</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    # Derive dropdown options from the dataset
    job_titles       = sorted(df["job_title"].unique().tolist())
    education_levels = ["High School", "Diploma", "Bachelor", "Master", "PhD"]
    education_levels = [e for e in education_levels if e in df["education_level"].unique()]
    industries       = sorted(df["industry"].unique().tolist())
    company_sizes    = sorted(df["company_size"].unique().tolist())
    locations        = sorted(df["location"].unique().tolist())
    remote_options   = sorted(df["remote_work"].unique().tolist())

    with col1:
        job_title      = st.selectbox("🧑‍💻 Job Title", job_titles)
        education      = st.selectbox("🎓 Education Level", education_levels)
        industry       = st.selectbox("🏭 Industry", industries)

    with col2:
        company_size   = st.selectbox("🏢 Company Size", company_sizes)
        location       = st.selectbox("📍 Location", locations)
        remote_work    = st.selectbox("🌐 Remote Work", remote_options)

    with col3:
        experience     = st.slider("📅 Years of Experience", 0, 25, 5)
        skills_count   = st.slider("🛠️  Skills Count", 0, 20, 8)
        certifications = st.slider("🏅 Certifications", 0, 10, 2)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮  Predict Salary", type="primary", use_container_width=True)

    if predict_btn:
        if not model_ready:
            st.error("Model is not loaded. Please train the model first.")
        else:
            input_dict = {
                "job_title":        job_title,
                "education_level":  education,
                "industry":         industry,
                "company_size":     company_size,
                "location":         location,
                "remote_work":      remote_work,
                "experience_years": experience,
                "skills_count":     skills_count,
                "certifications":   certifications,
            }
            row_enc   = encode_single_row(input_dict, encoders)
            predicted = model.predict(row_enc)[0]

            st.markdown(
                f"""
                <div class="prediction-box">
                    <p>Estimated Annual Salary</p>
                    <h1>${predicted:,.0f}</h1>
                    <p>Model used: <b>{best_name}</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Show input summary
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">Input Summary</div>', unsafe_allow_html=True)
            summary_df = pd.DataFrame([input_dict]).T
            summary_df.columns = ["Value"]
            summary_df.index.name = "Feature"
            st.dataframe(summary_df, use_container_width=True)

    # ── Quick stats strip ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">Dataset Quick Stats</div>', unsafe_allow_html=True)
    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("📋 Total Records",   f"{len(df):,}")
    q2.metric("💰 Avg Salary",      f"${df['salary'].mean():,.0f}")
    q3.metric("📉 Min Salary",      f"${df['salary'].min():,.0f}")
    q4.metric("📈 Max Salary",      f"${df['salary'].max():,.0f}")
    q5.metric("👔 Job Titles",      df["job_title"].nunique())


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 2 – Dataset Overview
# ════════════════════════════════════════════════════════════════════════════════
elif active == "overview":
    st.title("📊 Dataset Overview")

    tab1, tab2, tab3 = st.tabs(["Preview", "Statistics", "Value Counts"])

    with tab1:
        st.markdown('<div class="section-header">First 100 rows</div>', unsafe_allow_html=True)
        st.dataframe(df.head(100), use_container_width=True)

    with tab2:
        st.markdown('<div class="section-header">Descriptive Statistics</div>', unsafe_allow_html=True)
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    with tab3:
        st.markdown('<div class="section-header">Categorical Column Value Counts</div>', unsafe_allow_html=True)
        cat_col = st.selectbox(
            "Choose column",
            ["job_title", "education_level", "industry", "company_size", "location", "remote_work"],
        )
        vc = df[cat_col].value_counts().reset_index()
        vc.columns = [cat_col, "count"]
        c1, c2 = st.columns([1, 2])
        c1.dataframe(vc, use_container_width=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=vc.head(15), x="count", y=cat_col, ax=ax, palette="Blues_d")
        ax.set_title(f"Top values — {cat_col}")
        plt.tight_layout()
        c2.pyplot(fig)
        plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 3 – EDA Visualizations
# ════════════════════════════════════════════════════════════════════════════════
elif active == "eda":
    st.title("📈 Exploratory Data Analysis")

    plots = [
        ("salary_distribution.png",   "Salary Distribution"),
        ("avg_salary_by_job.png",      "Average Salary by Job Title"),
        ("salary_vs_experience.png",   "Salary vs Experience"),
        ("avg_salary_by_education.png","Average Salary by Education Level"),
        ("avg_salary_by_remote.png",   "Average Salary by Remote Work"),
        ("correlation_heatmap.png",    "Correlation Heatmap"),
    ]

    any_found = False
    for i in range(0, len(plots), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j >= len(plots):
                break
            fname, title = plots[i + j]
            fpath = asset(fname)
            if os.path.exists(fpath):
                col.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
                col.image(fpath, use_container_width=True)
                any_found = True
            else:
                col.info(f"Plot not found: {fname}. Run model_training.py first.")

    if not any_found:
        st.warning("No EDA plots found. Please run `python model_training.py` first.")


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 4 – Model Performance
# ════════════════════════════════════════════════════════════════════════════════
elif active == "model":
    st.title("🤖 Model Performance")

    if results is None:
        st.warning("No results found. Please run `python model_training.py` first.")
    else:
        # Metrics table
        st.markdown('<div class="section-header">Evaluation Metrics</div>', unsafe_allow_html=True)
        df_res = pd.DataFrame(results).set_index("model")
        df_res.columns = ["MAE ($)", "MSE ($²)", "RMSE ($)", "R² Score"]
        df_res = df_res.style.format(
            {"MAE ($)": "{:,.2f}", "MSE ($²)": "{:,.2f}", "RMSE ($)": "{:,.2f}", "R² Score": "{:.4f}"}
        ).highlight_max(subset=["R² Score"], color="#d4edda") \
         .highlight_min(subset=["RMSE ($)", "MAE ($)"], color="#d4edda")
        st.dataframe(df_res, use_container_width=True)

        st.markdown(
            f"""
            <div class="info-banner">
            ✅  <b>Best model selected: {best_name}</b><br>
            The model with the highest R² score and lowest RMSE was automatically saved.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Model comparison chart
        st.markdown("<br>", unsafe_allow_html=True)
        cmp_path = asset("model_comparison.png")
        if os.path.exists(cmp_path):
            st.markdown('<div class="section-header">Performance Comparison Chart</div>', unsafe_allow_html=True)
            st.image(cmp_path, use_container_width=True)

        # Feature importance
        fi_path = asset("feature_importance.png")
        if os.path.exists(fi_path):
            st.markdown('<div class="section-header">Feature Importance (Best Model)</div>', unsafe_allow_html=True)
            st.image(fi_path, use_container_width=True)
        else:
            st.info("Feature importance chart is only available for tree-based models (Random Forest / Gradient Boosting).")

        # Metric explanations
        st.markdown("---")
        st.markdown('<div class="section-header">What do the metrics mean?</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.markdown(
            """
            **MAE — Mean Absolute Error**
            Average absolute difference between predicted and actual salary.
            Lower is better.

            **MSE — Mean Squared Error**
            Average of squared differences. Penalises large errors more.
            Lower is better.
            """
        )
        c2.markdown(
            """
            **RMSE — Root Mean Squared Error**
            Square root of MSE — same unit as salary (USD).
            Lower is better.

            **R² Score**
            Proportion of salary variance explained by the model.
            Closer to 1.0 is better.
            """
        )
