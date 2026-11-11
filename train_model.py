"""Train and compare regression models for the House Price Predictor."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "house_prices.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
FEATURES = ["area_sqft", "bedrooms", "bathrooms", "age_years", "location_score", "parking_spaces"]
TARGET = "price_inr"

def train_and_save():
    data = pd.read_csv(DATA_PATH).drop_duplicates()
    data[FEATURES + [TARGET]] = data[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=[TARGET])
    X, y = data[FEATURES], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    candidates = {
        "Linear Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=220, random_state=42, min_samples_leaf=2,
                n_jobs=-1
            ))
        ])
    }

    results = {}
    best_name, best_model, best_r2 = None, None, float("-inf")
    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        metrics = {
            "MAE": float(mean_absolute_error(y_test, pred)),
            "RMSE": float(mean_squared_error(y_test, pred) ** 0.5),
            "R2": float(r2_score(y_test, pred))
        }
        results[name] = metrics
        if metrics["R2"] > best_r2:
            best_name, best_model, best_r2 = name, pipeline, metrics["R2"]

    joblib.dump(best_model, MODEL_DIR / "house_price_model.joblib")
    payload = {
        "best_model": best_name,
        "metrics": results,
        "features": FEATURES,
        "dataset_rows": int(len(data)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test))
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(payload, indent=2))
    pd.DataFrame(results).T.to_csv(BASE_DIR / "model_evaluation.csv")
    print(pd.DataFrame(results).T.round(3))
    print(f"\nBest model saved: {best_name}")
    return best_model

if __name__ == "__main__":
    train_and_save()
