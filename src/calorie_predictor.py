"""Predicts typical daily calorie intake from body measurements.

Wraps the Random Forest trained by scripts/train_models.py on NHANES 2017-2018.

This is descriptive, not prescriptive: it answers "how much do people with this
age, sex and build actually eat", learned from measured dietary recalls. It is
not a recommendation. calculations.calculate_daily_calorie_target stays the
number the app plans meals against; this sits beside it as context.

It ships because it beat the Mifflin-St Jeor formula on held-out data
(MAE 632.9 vs 650.6 kcal). That margin is small and R2 is only 0.130 — a single
24-hour recall is a noisy measure of habitual intake, so most of the variance is
not predictable from body measurements by any model. Exact figures are written
to models/model_metrics.json at training time.
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
        """Held-out test metrics recorded at training time, if available."""
        return self._metrics

    def predict(
        self,
        age_years: int,
        is_male: bool,
        height_cm: float,
        weight_kg: float,
        bmi: float,
    ) -> float:
        """Typical daily calorie intake, in kcal, for this profile.

        Raises:
            RuntimeError: If the model could not be loaded.
            ValueError: If any measurement is not positive.
        """
        if not self.is_available:
            raise RuntimeError(f"Calorie model unavailable: {self._load_error}")

        if min(age_years, height_cm, weight_kg, bmi) <= 0:
            raise ValueError("age, height, weight and BMI must all be positive")

        features = pd.DataFrame(
            [[age_years, int(is_male), height_cm, weight_kg, bmi]],
            columns=FEATURE_COLUMNS,
        )
        return round(float(self._model.predict(features)[0]), 0)
