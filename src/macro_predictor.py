"""Predicts daily macro targets (protein/carbs/fat/sugar) from a person's
height, weight and calorie target, using the trained Random Forest model.

This is the one place the app depends on a model file rather than a plain
formula - there's no simple closed-form equation for "how much protein
should this specific body need", so it's a reasonable place to let a model
trained on real data do the work instead of guessing at a rule.
"""

from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

INPUT_COLUMNS = ["Height", "Weight", "Food Energy (Calories/day)"]


class MacroPredictor:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self._model = None
        self._scaler_x = None
        self._scaler_y = None
        self._load_error = None
        try:
            self._model = joblib.load(models_dir / "random_forest_regressor.pkl")
            self._scaler_x = joblib.load(models_dir / "scaler_X.pkl")
            self._scaler_y = joblib.load(models_dir / "scaler_y.pkl")
        except (FileNotFoundError, OSError, EOFError) as exc:
            self._load_error = str(exc)

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def predict(self, height_cm: float, weight_kg: float, calories: float) -> dict:
        if not self.is_available:
            raise RuntimeError(f"Macro prediction model unavailable: {self._load_error}")

        input_df = pd.DataFrame([[height_cm, weight_kg, calories]], columns=INPUT_COLUMNS)
        input_scaled = self._scaler_x.transform(input_df)
        prediction_scaled = self._model.predict(input_scaled)
        protein, carbs, fat, sugar = self._scaler_y.inverse_transform(prediction_scaled)[0]

        return {
            "protein_g": round(float(protein), 1),
            "carbs_g": round(float(carbs), 1),
            "fat_g": round(float(fat), 1),
            "sugar_g": round(float(sugar), 1),
        }
