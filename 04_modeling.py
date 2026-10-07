import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

FEATURED_PATH = "data/insurance_featured.csv"
OUT_DIR = "outputs"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
RANDOM_STATE = 42

NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region", "bmi_category", "age_group", "smoker_obese"]


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}


def main():
    df = pd.read_csv(FEATURED_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor()
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=RANDOM_STATE),
    }

    results = []
    fitted = {}
    for name, reg in models.items():
        pipe = Pipeline([("prep", preprocessor), ("reg", reg)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        results.append(evaluate(name, y_test, y_pred))
        fitted[name] = pipe

    comparison = pd.DataFrame(results).sort_values("rmse")
    comparison.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)
    print("\n=== Model comparison (sorted by RMSE - lower is better) ===")
    print(comparison.to_string(index=False))

    best_name = comparison.iloc[0]["model"]
    best_pipe = fitted[best_name]
    print(f"\nBest model: {best_name}")
    joblib.dump(best_pipe, f"{MODEL_DIR}/best_model.pkl")

    y_pred_best = best_pipe.predict(X_test)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred_best, alpha=0.4)
    lims = [0, max(y_test.max(), y_pred_best.max())]
    ax.plot(lims, lims, "r--", alpha=0.6)
    ax.set_xlabel("Actual charges ($)")
    ax.set_ylabel("Predicted charges ($)")
    ax.set_title(f"Actual vs predicted - {best_name}")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/actual_vs_predicted.png")
    plt.close(fig)

    best_rmse = comparison.iloc[0]["rmse"]
    best_mae = comparison.iloc[0]["mae"]
    best_r2 = comparison.iloc[0]["r2"]
    print(f"\nOn held-out data, the best model is off by about ${best_mae:,.0f} "
          f"on average (MAE), explains {best_r2:.1%} of the variance in "
          f"charges (R²), with an RMSE of ${best_rmse:,.0f} (penalizes larger "
          f"misses more than MAE does).")

    print(f"\nSaved comparison table and plot to {OUT_DIR}/, model to {MODEL_DIR}/best_model.pkl")


if __name__ == "__main__":
    main()
