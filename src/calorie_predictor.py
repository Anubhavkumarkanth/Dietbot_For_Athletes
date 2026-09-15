"""Loads the calorie model trained by scripts/train_models.py.

Descriptive, not prescriptive: it estimates what people with a similar build
actually eat, which is not the same as what someone should eat. The app plans
meals from the Mifflin-St Jeor target and shows this beside it.

Metrics are in models/model_metrics.json.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# Must match FEATURES in scripts/train_models.py, in the same order.
FEATURE_COLUMNS = ["age_years", "SEX_MALE", "height_cm", "weight_kg", "bmi"]


class CaloriePredictor:
    """Loads the trained model, or degrades gracefully if it is missing.

    Construction never raises: the app checks is_available and omits the
    reference figure rather than failing to start over an optional feature.
    """

    def __init__(self, models_dir: Path = MODELS_DIR):
        self._model = None
        self._metrics = None
        self._load_error = None

        try:
            self._model = joblib.load(models_dir / "calorie_intake_rf.pkl")
        except (FileNotFoundError, OSError, EOFError) as exc:
            self._load_error = str(exc)
            return

        try:
            self._metrics = json.loads(
                (models_dir / "model_metrics.json").read_text(encoding="utf-8")
            )
        except (FileNotFoundError, OSError, ValueError):
            # Display only - a usable model without them is fine.
            self._metrics = None

    @property
    def is_available(self) -> bool:
        return self._model is not None

    @property
    def metrics(self) -> dict | None:
        return self._metrics

    def predict(
        self,
        age_years: int,
        is_male: bool,
        height_cm: float,
        weight_kg: float,
        bmi: float,
    ) -> float:
        """Typical daily calorie intake, in kcal, for this profile."""
        if not self.is_available:
            raise RuntimeError(f"Calorie model unavailable: {self._load_error}")

        if min(age_years, height_cm, weight_kg, bmi) <= 0:
            raise ValueError("age, height, weight and BMI must all be positive")

        features = pd.DataFrame(
            [[age_years, int(is_male), height_cm, weight_kg, bmi]],
            columns=FEATURE_COLUMNS,
        )
        return round(float(self._model.predict(features)[0]), 0)
