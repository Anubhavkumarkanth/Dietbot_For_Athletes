# DietBot

A Streamlit app that generates a personalized daily macro target and meal plan for athletes, based on their body stats, sport, and training load.

## Why I built it

Most generic diet calculators ignore the fact that a swimmer training for two hours has very different needs than someone doing a 20-minute yoga session, even at the same height and weight. I wanted something that took activity type and duration into account, backed by an actual dataset of macro requirements rather than a single one-size-fits-all formula, and paired with real food data instead of a generic "eat more protein" suggestion.

## Features

- Predicts daily protein/carbs/fat/sugar targets from height, weight, and a computed daily calorie target (via a trained Random Forest model)
- Calculates BMI, BMR, TDEE, and session calorie burn using standard formulas — no ML needed for these
- Builds meal plans by matching real foods to your remaining macro needs, using a Nearest Neighbors model over a 145-item food dataset
- Generates several meal plan variations per request
- Optional AI-written plain-language summary of your plan (falls back to a templated summary if no API key is set)
- Streamlit UI — fill in your stats, click a button, get your plan

## How it works

```
User input (height, weight, age, gender, sport, duration, activity level, goal)
        │
        ├─► Deterministic calculations (BMI, BMR, TDEE, session calories burned)
        │
        ▼
Daily calorie target
        │
        ▼
Random Forest model → daily macro targets (protein/carbs/fat/sugar)
        │
        ▼
Nearest Neighbors model → matches foods to remaining macro needs, builds meal plans
        │
        ▼
Optional AI layer → turns the numbers into a short natural-language summary
```

## Tech stack

- Python
- Streamlit — UI
- scikit-learn — Random Forest (macro prediction) and Nearest Neighbors (food matching)
- pandas / NumPy — data handling
- OpenAI API — optional plan summaries (the app works fully without it)

## Project structure

```
DietBot/
├── app.py                    # Streamlit app
├── requirements.txt
├── .env.example
├── data/
│   ├── food_data.csv         # 145 foods with full nutrient breakdowns
│   └── macro_targets.xlsx    # training data: height/weight → macro targets
├── models/                   # trained model files (committed, ready to use)
├── src/
│   ├── calculations.py       # BMI, BMR/TDEE, MET-based calorie burn — no ML
│   ├── macro_predictor.py    # loads the Random Forest model
│   ├── meal_recommender.py   # loads the Nearest Neighbors model, builds meal plans
│   └── ai_explainer.py       # optional OpenAI-based plan summary, with fallback
├── scripts/
│   └── train_models.py       # regenerates the .pkl files in models/
├── notebooks/
│   └── training.ipynb        # the same training process, with exploratory plots
└── tests/
    ├── test_calculations.py
    └── test_models.py
```

## Installation

```bash
git clone https://github.com/Anubhavkumarkanth/Dietbot-An-AI-Powered-Diet-Plans-for-Atheletes.git
cd Dietbot-An-AI-Powered-Diet-Plans-for-Atheletes
pip install -r requirements.txt
```

## Environment variables

Copy `.env.example` to `.env` and fill in what you want to use:

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | No | Enables AI-generated plan summaries. Without it, the app shows a templated summary instead. |

## Running the project

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

To retrain the models from scratch instead of using the committed `.pkl` files:

```bash
python scripts/train_models.py
```

## Example usage

Fill in your stats in the sidebar — say, 182cm, 75kg, 25 years old, male, Basketball for 60 minutes, Moderately Active, goal: Muscle Gain — and click **Get recommendations**. You'll get:

- BMI, session calories burned, and a daily calorie target
- Daily protein/carbs/fat/sugar targets
- A few meal plan options, e.g. one built around *Idli*, *Dosa*, and *Egg burrito*, with totals shown for each
- A short summary of the plan in plain language

## AI usage

This project deliberately keeps AI to the one place it actually helps: turning a finished set of numbers into a short, readable summary.

**Handled without AI:**
- BMI, BMR, TDEE, and session calorie burn — standard formulas
- Daily macro targets — a Random Forest model trained on the height/weight/calorie dataset
- Meal plan construction — a Nearest Neighbors model matching foods to the remaining macro gap

**Handled with AI (optional):**
- The natural-language plan summary shown at the bottom of the results, generated via the OpenAI API

If `OPENAI_API_KEY` isn't set, or the API call fails or times out, the app falls back to a templated summary built from the same numbers — the core functionality never depends on the AI service being available.

## Limitations

- The macro-prediction model is trained on a small synthetic dataset (~100 rows), so treat its output as a reasonable starting point, not medical advice.
- The food dataset has 145 items, mostly South Asian dishes — it won't cover every cuisine.
- Meal plans are built with a greedy nearest-neighbor search, not a true optimizer, so they get close to your macro targets rather than hitting them exactly.
- No allergy, dietary restriction (vegetarian/vegan/halal/etc.), or meal-timing support yet.
- No persistence — nothing is saved between sessions.

## Future improvements

- Filter meal plans by dietary restrictions (vegetarian, allergies, etc.)
- Swap the greedy meal-matching for a proper constrained optimizer (e.g. linear programming) to hit macro targets more precisely
- Expand the food dataset, or pull from a public nutrition API instead of a static CSV
- Let users save and revisit past plans
