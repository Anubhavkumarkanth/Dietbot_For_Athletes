"""Builds meal plans out of the food dataset using the trained
NearestNeighbors model.

For each meal plan, it repeatedly asks the model for foods close to
whatever macros are still missing, adds one, and subtracts it from the
remaining target - a simple greedy fill rather than an exhaustive search,
which is fine here since we only need a handful of reasonable plans, not
a perfectly optimal one.
"""

from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Maps the raw food-dataset column names to the friendlier names used
# everywhere else in the app.
FOOD_COLUMNS = {
    "Food_name": "food_name",
    "Protein(g)": "protein",
    "Total lipid (fat)(g)": "fat",
    "Carbohydrate, by difference(g)": "carbs",
}


class MealRecommender:
    def __init__(self, models_dir: Path = MODELS_DIR, data_dir: Path = DATA_DIR):
        self._foods = None
        self._nn_model = None
        self._load_error = None
        try:
            food_data = pd.read_csv(data_dir / "food_data.csv", encoding="latin1")
            self._foods = (
                food_data[list(FOOD_COLUMNS)]
                .dropna()
                .rename(columns=FOOD_COLUMNS)
                .reset_index(drop=True)
            )
            self._nn_model = joblib.load(models_dir / "nearest_neighbors_model.pkl")
        except (FileNotFoundError, OSError, EOFError, KeyError) as exc:
            self._load_error = str(exc)

    @property
    def is_available(self) -> bool:
        return self._nn_model is not None and self._foods is not None

    def recommend_meal_plans(
        self,
        protein_target: float,
        fat_target: float,
        carbs_target: float,
        plan_count: int = 3,
        max_items: int = 5,
    ) -> list[dict]:
        if not self.is_available:
            raise RuntimeError(f"Meal recommender unavailable: {self._load_error}")

        plans = []
        for _ in range(plan_count):
            plans.append(
                self._build_one_plan(protein_target, fat_target, carbs_target, max_items)
            )
        return plans

    def _build_one_plan(self, protein_target, fat_target, carbs_target, max_items):
        remaining = {"protein": protein_target, "fat": fat_target, "carbs": carbs_target}
        used_indices = set()
        meals = []

        neighbor_count = min(5, len(self._foods))

        for _ in range(max_items):
            query = [[
                max(remaining["protein"], 0),
                max(remaining["fat"], 0),
                max(remaining["carbs"], 0),
            ]]
            _, indices = self._nn_model.kneighbors(query, n_neighbors=neighbor_count)

            candidate = next((idx for idx in indices[0] if idx not in used_indices), None)
            if candidate is None:
                break

            used_indices.add(candidate)
            food = self._foods.iloc[candidate]
            meals.append({
                "food_name": food["food_name"],
                "protein": round(float(food["protein"]), 1),
                "fat": round(float(food["fat"]), 1),
                "carbs": round(float(food["carbs"]), 1),
            })

            remaining["protein"] -= food["protein"]
            remaining["fat"] -= food["fat"]
            remaining["carbs"] -= food["carbs"]

            if remaining["protein"] <= 0 and remaining["fat"] <= 0 and remaining["carbs"] <= 0:
                break

        totals = {
            "protein": round(sum(m["protein"] for m in meals), 1),
            "fat": round(sum(m["fat"] for m in meals), 1),
            "carbs": round(sum(m["carbs"] for m in meals), 1),
        }
        return {"meals": meals, "totals": totals}
