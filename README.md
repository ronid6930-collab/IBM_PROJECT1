# 💼 Job Salary Prediction

A beginner-friendly Machine Learning project that predicts employee salaries
based on job title, education, industry, experience, and more.

---

## 📁 Project Structure

```
salary_prediction/
├── data_processing.py   ← Load, clean, encode the dataset
├── model_training.py    ← Train & evaluate ML models, save artefacts
├── app.py               ← Streamlit web application
├── requirements.txt     ← Python dependencies
├── models/              ← Saved model & encoders (created after training)
│   ├── best_model.pkl
│   ├── encoders.pkl
│   ├── results.json
│   └── best_model_name.txt
└── assets/              ← Saved plots (created after training)
    ├── salary_distribution.png
    ├── avg_salary_by_job.png
    ├── salary_vs_experience.png
    ├── avg_salary_by_education.png
    ├── avg_salary_by_remote.png
    ├── correlation_heatmap.png
    ├── model_comparison.png
    └── feature_importance.png

job_salary_prediction_dataset.csv   ← Raw dataset (250,000 rows)
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
cd salary_prediction
pip install -r requirements.txt
```

### 2. Train the model

```bash
python model_training.py
```

This will:
- Load and clean the dataset
- Generate EDA plots → `assets/`
- Train 3 regression models
- Print evaluation metrics (MAE, MSE, RMSE, R²)
- Save the best model → `models/best_model.pkl`

### 3. Launch the Streamlit app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📊 Dataset

| Column             | Type        | Description                              |
|--------------------|-------------|------------------------------------------|
| `job_title`        | Categorical | Role name (AI Engineer, Data Analyst …)  |
| `experience_years` | Numeric     | Years of work experience (0–25)          |
| `education_level`  | Categorical | Highest education (High School → PhD)    |
| `skills_count`     | Numeric     | Number of skills listed                  |
| `industry`         | Categorical | Industry sector                          |
| `company_size`     | Categorical | Startup / Small / Medium / Large / Enterprise |
| `location`         | Categorical | Country or Remote                        |
| `remote_work`      | Categorical | Yes / No / Hybrid                        |
| `certifications`   | Numeric     | Number of professional certifications    |
| `salary`           | Numeric     | **Target** — Annual salary in USD        |

---

## 🤖 Models Trained

| Model                     | Notes                                   |
|---------------------------|-----------------------------------------|
| Linear Regression         | Simple baseline, fast to train          |
| Random Forest Regressor   | Ensemble, handles non-linearity well    |
| Gradient Boosting Regressor | Boosted trees, usually highest accuracy |

The model with the best **R² score** is automatically selected and saved.

---

## 🖥️ Streamlit App Pages

| Page                  | What you'll find                                          |
|-----------------------|-----------------------------------------------------------|
| 🏠 Home & Predict     | Input form + instant salary prediction                    |
| 📊 Dataset Overview   | Preview, descriptive statistics, value counts             |
| 📈 EDA Visualizations | 6 charts: distribution, by job/education/remote, heatmap  |
| 🤖 Model Performance  | Metrics table, comparison chart, feature importance       |

---

## 📖 Key Concepts (for Viva)

- **Label Encoding** — converts text categories to integers for ML models
- **Train/Test Split** — 80% for training, 20% for evaluating generalization
- **R² Score** — how well the model explains salary variance (1.0 = perfect)
- **RMSE** — average prediction error in USD; lower is better
- **Feature Importance** — which inputs most influence salary (tree models only)
- **Joblib** — used to save/load the trained model without retraining

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Pandas / NumPy** — data manipulation
- **Matplotlib / Seaborn** — visualizations
- **Scikit-learn** — ML models & metrics
- **Streamlit** — web UI
- **Joblib** — model persistence
