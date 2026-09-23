"""
model_training.py
-----------------
Trains Linear Regression, Random Forest Regressor, and Gradient Boosting
Regressor on the salary dataset, evaluates them, and saves the best model
along with the encoders to the models/ directory.

Run this script once before launching the Streamlit app:
    python model_training.py
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ── Local imports ─────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_processing import (
    run_pipeline,
    save_encoders,
    MODELS_DIR,
    FEATURE_COLS,
)

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── Evaluation helper ─────────────────────────────────────────────────────────
def evaluate(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    print(f"\n{'-'*40}")
    print(f"  {name}")
    print(f"  MAE  : {mae:>12,.2f}")
    print(f"  MSE  : {mse:>12,.2f}")
    print(f"  RMSE : {rmse:>12,.2f}")
    print(f"  R2   : {r2:>12.4f}")
    return {"model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


# ── EDA Plots ─────────────────────────────────────────────────────────────────
def save_eda_plots(df: pd.DataFrame):
    """Generate and save EDA plots to the assets/ directory."""
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Salary Distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["salary"], bins=50, kde=True, ax=ax, color="#3b82d4")
    ax.set_title("Salary Distribution")
    ax.set_xlabel("Salary (USD)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "salary_distribution.png"), dpi=100)
    plt.close(fig)

    # 2. Average Salary by Job Title
    avg_by_job = (
        df.groupby("job_title")["salary"]
        .mean()
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    avg_job_df = avg_by_job.reset_index()
    avg_job_df.columns = ["job_title", "salary"]
    sns.barplot(data=avg_job_df, x="salary", y="job_title", hue="job_title",
                ax=ax, palette="Blues_d", legend=False)
    ax.set_title("Average Salary by Job Title")
    ax.set_xlabel("Average Salary (USD)")
    ax.set_ylabel("Job Title")
    plt.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "avg_salary_by_job.png"), dpi=100)
    plt.close(fig)

    # 3. Salary vs Experience (scatter)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(
        data=df.sample(min(5000, len(df)), random_state=42),
        x="experience_years", y="salary",
        alpha=0.4, ax=ax, color="#7c5cd8",
    )
    ax.set_title("Salary vs Experience Years")
    ax.set_xlabel("Experience (years)")
    ax.set_ylabel("Salary (USD)")
    plt.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "salary_vs_experience.png"), dpi=100)
    plt.close(fig)

    # 4. Average Salary by Education Level
    edu_order = ["High School", "Diploma", "Bachelor", "Master", "PhD"]
    edu_order = [e for e in edu_order if e in df["education_level"].unique()]
    avg_edu = df.groupby("education_level")["salary"].mean().reindex(edu_order)
    fig, ax = plt.subplots(figsize=(8, 4))
    avg_edu_df = avg_edu.reset_index()
    avg_edu_df.columns = ["education_level", "salary"]
    sns.barplot(data=avg_edu_df, x="education_level", y="salary", hue="education_level",
                ax=ax, palette="Greens_d", legend=False)
    ax.set_title("Average Salary by Education Level")
    ax.set_xlabel("Education Level")
    ax.set_ylabel("Average Salary (USD)")
    plt.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "avg_salary_by_education.png"), dpi=100)
    plt.close(fig)

    # 5. Average Salary by Remote Work
    fig, ax = plt.subplots(figsize=(6, 4))
    df.groupby("remote_work")["salary"].mean().plot(kind="bar", ax=ax, color=["#3b82d4","#7c5cd8","#e5e7eb"])
    ax.set_title("Average Salary by Remote Work Type")
    ax.set_xlabel("Remote Work")
    ax.set_ylabel("Average Salary (USD)")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "avg_salary_by_remote.png"), dpi=100)
    plt.close(fig)

    # 6. Correlation heatmap (numerical columns only)
    num_cols = ["experience_years", "skills_count", "certifications", "salary"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        df[num_cols].corr(),
        annot=True, fmt=".2f", cmap="coolwarm",
        ax=ax, linewidths=0.5,
    )
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "correlation_heatmap.png"), dpi=100)
    plt.close(fig)

    print("[OK] EDA plots saved to assets/")


# ── Model Performance Plot ────────────────────────────────────────────────────
def save_metrics_plot(results: list[dict]):
    names  = [r["model"] for r in results]
    r2s    = [r["R2"]    for r in results]
    rmses  = [r["RMSE"]  for r in results]

    x = np.arange(len(names))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars1 = ax1.bar(x, r2s, width, color=["#3b82d4", "#7c5cd8", "#22c55e"])
    ax1.set_title("R² Score Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha="right")
    ax1.set_ylabel("R² Score")
    ax1.set_ylim(0, 1)
    for bar in bars1:
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    bars2 = ax2.bar(x, rmses, width, color=["#3b82d4", "#7c5cd8", "#22c55e"])
    ax2.set_title("RMSE Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha="right")
    ax2.set_ylabel("RMSE (USD)")
    for bar in bars2:
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f"{bar.get_height():,.0f}",
            ha="center", va="bottom", fontsize=9,
        )

    plt.suptitle("Model Performance Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "model_comparison.png"), dpi=100)
    plt.close(fig)
    print("[OK] Model comparison plot saved.")


# ── Feature Importance Plot ───────────────────────────────────────────────────
def save_feature_importance(model, feature_names: list[str]):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_names  = [feature_names[i] for i in indices]
    sorted_values = importances[indices]

    fi_df = pd.DataFrame({"feature": sorted_names, "importance": sorted_values})
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=fi_df, x="importance", y="feature", hue="feature",
                ax=ax, palette="viridis", legend=False)
    ax.set_title("Feature Importance (Best Model)")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "feature_importance.png"), dpi=100)
    plt.close(fig)
    print("[OK] Feature importance plot saved.")


# ── Main training routine ─────────────────────────────────────────────────────
def train():
    print("=" * 50)
    print("  JOB SALARY PREDICTION - MODEL TRAINING")
    print("=" * 50)

    # 1. Load & preprocess
    print("\n[1/5] Loading and preprocessing data ...")
    t0 = time.time()
    X, y, df_clean, encoders = run_pipeline()
    print(f"      Dataset shape : {df_clean.shape}")
    print(f"      Features      : {list(X.columns)}")
    print(f"      Done in {time.time()-t0:.1f}s")

    # 2. EDA plots
    print("\n[2/5] Generating EDA plots ...")
    save_eda_plots(df_clean)

    # 3. Train/test split
    print("\n[3/5] Splitting data (80 / 20) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"      Train: {X_train.shape[0]} rows  |  Test: {X_test.shape[0]} rows")

    # 4. Train models
    print("\n[4/5] Training models ...")
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
        ),
    }

    results = []
    trained = {}
    for name, model in models.items():
        print(f"  Training {name} ...", end="", flush=True)
        t = time.time()
        model.fit(X_train, y_train)
        print(f" done ({time.time()-t:.1f}s)")
        r = evaluate(name, model, X_test, y_test)
        results.append(r)
        trained[name] = model

    # 5. Select best model (highest R²)
    best_result = max(results, key=lambda r: r["R2"])
    best_name   = best_result["model"]
    best_model  = trained[best_name]
    print(f"\n\n[BEST] Best model : {best_name}  (R2 = {best_result['R2']:.4f})")

    # 6. Save artefacts
    print("\n[5/5] Saving artefacts ...")
    model_path    = os.path.join(MODELS_DIR, "best_model.pkl")
    encoders_path = os.path.join(MODELS_DIR, "encoders.pkl")
    results_path  = os.path.join(MODELS_DIR, "results.json")
    best_name_path = os.path.join(MODELS_DIR, "best_model_name.txt")

    joblib.dump(best_model, model_path)
    save_encoders(encoders, encoders_path)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    with open(best_name_path, "w") as f:
        f.write(best_name)

    save_metrics_plot(results)
    save_feature_importance(best_model, FEATURE_COLS)

    print(f"\n[OK] Model saved    : {model_path}")
    print(f"[OK] Encoders saved : {encoders_path}")
    print(f"[OK] Results saved  : {results_path}")
    print("\n[DONE] Training complete! Run the Streamlit app with:")
    print("    streamlit run app.py\n")


if __name__ == "__main__":
    train()
