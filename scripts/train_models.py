"""Retrains the macro-prediction and food-matching models from the files in
`data/` and writes fresh pickles to `models/`.

The trained models are already committed to the repo, so you don't need to
run this to use the app. Run it if you change the training data, want to
reproduce the shipped models yourself, or bump the scikit-learn version:

    python scripts/train_models.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

INPUT_FEATURES = ["Height", "Weight", "Food Energy (Calories/day)"]
OUTPUT_FEATURES = [
    "Protein (grams/day)",
    "Carbs (grams/day)",
    "Fat (grams/day)",
    "Sugar (grams/day)",
]

FOOD_COLUMNS = {
    "Food_name": "food_name",
    "Protein(g)": "protein",
    "Total lipid (fat)(g)": "fat",
    "Carbohydrate, by difference(g)": "carbs",
}


def train_macro_model() -> None:
    df = pd.read_excel(DATA_DIR / "macro_targets.xlsx", sheet_name="Sheet1")
    df.columns = [c.strip() for c in df.columns]
    df["Food Energy (Calories/day)"] = (
        df["Food Energy (Calories/day)"].astype(str).str.replace(",", "").astype(float)
    )
    df = df.dropna(subset=INPUT_FEATURES + OUTPUT_FEATURES)

    X = df[INPUT_FEATURES]
    y = df[OUTPUT_FEATURES]

    scaler_x = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y)
    X_scaled = scaler_x.transform(X)
    y_scaled = scaler_y.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(
        f"Macro model  MSE={mean_squared_error(y_test, y_pred):.3f}  "
        f"R2={r2_score(y_test, y_pred):.3f}  (n={len(df)})"
    )

    joblib.dump(model, MODELS_DIR / "random_forest_regressor.pkl")
    joblib.dump(scaler_x, MODELS_DIR / "scaler_X.pkl")
    joblib.dump(scaler_y, MODELS_DIR / "scaler_y.pkl")


def train_food_matcher() -> None:
    food_data = pd.read_csv(DATA_DIR / "food_data.csv", encoding="latin1")
    foods = (
        food_data[list(FOOD_COLUMNS)]
        .dropna()
        .rename(columns=FOOD_COLUMNS)
        .reset_index(drop=True)
    )

    model = NearestNeighbors(n_neighbors=5, metric="euclidean")
    model.fit(foods[["protein", "fat", "carbs"]].to_numpy())

    joblib.dump(model, MODELS_DIR / "nearest_neighbors_model.pkl")
    print(f"Food matcher trained on {len(foods)} food items")


if __name__ == "__main__":
    MODELS_DIR.mkdir(exist_ok=True)
    train_macro_model()
    train_food_matcher()
    print("Done. Models written to", MODELS_DIR)
