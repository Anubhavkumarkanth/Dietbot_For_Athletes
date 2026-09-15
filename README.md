# DietBot

A Streamlit app that turns a person's profile — height, weight, age, sex, sport, training duration
and goal — into a daily calorie target, macro targets, and a meal plan built from a nutrition
dataset. Plans can optionally be saved to PostgreSQL and read back later.

---

## Features

- Calorie target from the Mifflin-St Jeor equation, adjusted for activity level and goal
- A second calorie figure predicted by a Random Forest trained on NHANES survey data, shown as a
  reference for what people with a similar profile actually eat
- Daily protein, carb, fat and sugar targets derived from the calorie target
- Meal plans assembled by nearest-neighbour matching over a 145-item food table
- Optional plan history stored in PostgreSQL
- Optional plain-language summary via the OpenAI API, with a templated fallback

---

## How it works

```
profile input
   ├─ calculations.py      BMI, BMR, TDEE, session calorie burn
   │        ↓
   │   calorie target ──────────────┐
   │                                │
   ├─ calorie_predictor.py          │   Random Forest estimate of typical intake
   │   (reference figure only)      │   (shown beside the target, not used for the plan)
   │                                ↓
   ├─ macro_targets.py         protein / carbs / fat / sugar
   │                                ↓
   ├─ meal_recommender.py      nearest-neighbour food matching
   │                                ↓
   └─ storage.py               saved to PostgreSQL (optional)
```

The meal plan is always built from the Mifflin-St Jeor target. The model's number sits next to it
as context and does not drive the plan.

---

## ML approach

The app originally used a Random Forest to predict macro grams from height, weight and calories.
I removed it and replaced it with a different model, for a reason worth explaining.

### Why the original model was replaced

Its training data was 99 rows in which every macro target was a fixed multiple of the calorie
column:

| macro | grams per calorie | std | correlation with calories |
|---|---|---|---|
| protein | 0.060974 | 0.000141 | 0.999940 |
| carbs | 0.133316 | 0.000134 | 0.999988 |
| fat | 0.028419 | 0.000125 | 0.999769 |

A standard deviation in the fourth decimal place means those labels came from a formula, so the
model could only rediscover that formula. Worse, trees cannot extrapolate: above the 2,953 kcal
ceiling in the training data, **every input returned identical macros** — which covered most of the
athletes the app is aimed at.

### What replaced it

I switched to NHANES 2017–2018 (4,624 adults, US CDC, public domain) and tested four designs
against baselines. Three lost:

| Design | RF R² | Baseline | Kept |
|---|---|---|---|
| Macro grams from body + calories | 0.571 | 0.582 (calories × ratio) | No |
| Macro grams from body alone | 0.033 | −0.004 (mean) | No |
| Macro composition (% of energy) | −0.054 | −0.013 (mean) | No |
| **Calorie intake from body** | **0.130** | 0.071 (Mifflin-St Jeor) | **Yes** |

Run `python scripts/train_models.py --rejected` to reproduce the three failures.

The surviving model predicts daily calorie intake from age, sex, height, weight and BMI, trained
with a 60/20/20 split and `min_samples_leaf` chosen on validation only:

| Model | MAE (kcal) | R² |
|---|---|---|
| mean baseline | 684.4 | −0.001 |
| Mifflin-St Jeor | 650.6 | 0.071 |
| **Random Forest** | **632.9** | **0.130** |

Macro targets are now a direct calculation, because nothing beat the ratio there.

### Limitations

- **R² of 0.130 is low.** A single 24-hour dietary recall is a noisy measure of habitual intake
  (test-set standard deviation is about 855 kcal), so most of that variance is not predictable from
  body measurements by any model. The honest claim is narrow: modestly better than the standard
  formula at this task.
- It predicts **typical intake, not requirement**, and self-reported intake is known to be
  under-reported. It reads lower than the formula target.
- **Not medical advice.** No clinical claim is made.
- The macro split is fixed and does not vary by sport or goal — the data did not support varying it.
- The food table has 145 items, mostly South Asian dishes, so coverage is narrow.
- Meal plans use a greedy nearest-neighbour search, so they approach the macro targets rather than
  hitting them exactly.

---

## Data and attribution

**NHANES 2017–2018** — National Health and Nutrition Examination Survey, US CDC / National Center
for Health Statistics. Public domain. `scripts/build_dataset.py` downloads and joins the
demographics, body-measures and dietary-recall files; `data/nhanes_intake.csv` is the result.

**Original prototype** — this repository did not start empty. It began as a pair of notebooks
(`maincode.ipynb`, `model.ipynb`) with their `.pkl` artifacts and datasets, carrying a Colab badge
pointing at [SameerGoudageri/Final-Year-Project-](https://github.com/SameerGoudageri/Final-Year-Project-).
I did not write those and do not claim them. What I added: the `src/` rewrite, two correctness
fixes (the model was being fed a single workout's calorie burn instead of the daily target, and
`predict_nutrition` was called with its arguments in the wrong order), the NHANES pipeline and its
evaluation, the PostgreSQL layer, and the tests. The history is intact — commit `4363f9b` is the
original upload and `95c041b` is where the rewrite starts.

---

## Tech stack

Python · Streamlit · scikit-learn · pandas / NumPy · PostgreSQL (psycopg2) · pytest ·
OpenAI API (optional)

---

## Setup

```bash
git clone https://github.com/Anubhavkumarkanth/Dietbot-An-AI-Powered-Diet-Plans-for-Atheletes.git
cd Dietbot-An-AI-Powered-Diet-Plans-for-Atheletes
pip install -r requirements.txt
```

The app runs with no configuration. Both settings are optional:

```bash
cp .env.example .env
```

| Variable | Effect if unset |
|---|---|
| `OPENAI_API_KEY` | A templated plan summary is shown instead of a generated one |
| `DIETBOT_DB_URL` | Plans are not saved and no history is shown |

For persistence, create the database and load the schema:

```bash
psql -U postgres -c "CREATE ROLE dietbot_app LOGIN PASSWORD '<YOUR_PASSWORD>';"
psql -U postgres -c "CREATE DATABASE dietbot OWNER dietbot_app;"
psql -U dietbot_app -d dietbot -f sql/schema.sql
```

Then in `.env`:

```
DIETBOT_DB_URL=postgresql://dietbot_app:<YOUR_PASSWORD>@localhost:5432/dietbot
```

`.env` is gitignored and should never be committed.

---

## Running

```bash
streamlit run app.py
```

To rebuild the dataset or retrain (both artifacts are committed, so this is optional):

```bash
python scripts/build_dataset.py
python scripts/train_models.py
python scripts/train_models.py --rejected
```

---

## Tests

```bash
python -m pytest
```

24 tests. The five persistence tests skip automatically when `DIETBOT_DB_URL` is unset, so the
suite passes without a database.

They cover model loading, prediction type and range, that predictions respond to their inputs, a
regression test against the old model's flatline failure, invalid input handling, that the recorded
metrics still beat the formula baseline, macro targets reconciling with their calorie target, meal
plan construction, and — with a database — transactional saves, cascade deletes, and that a
constraint violation rolls the whole save back.

---

## Project structure

```
dietbot/
├── app.py                    Streamlit UI
├── sql/schema.sql            profiles → plans → plan_items
├── data/
│   ├── food_data.csv         145 foods with nutrient breakdowns
│   ├── nhanes_intake.csv     built by scripts/build_dataset.py
│   └── macro_targets.xlsx    the original dataset, kept for reference
├── models/
│   ├── calorie_intake_rf.pkl
│   ├── nearest_neighbors_model.pkl
│   └── model_metrics.json    held-out metrics, written at training time
├── src/
│   ├── calculations.py       BMI, BMR/TDEE, MET burn
│   ├── calorie_predictor.py  loads the Random Forest
│   ├── macro_targets.py      calorie target → macro grams
│   ├── meal_recommender.py   nearest-neighbour food matching
│   ├── storage.py            PostgreSQL persistence
│   └── ai_explainer.py       optional summary, with fallback
├── scripts/
│   ├── build_dataset.py
│   └── train_models.py
└── tests/
```
