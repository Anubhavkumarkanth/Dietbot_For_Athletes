# macro_model.py
import pickle
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import itertools
import random
import pandas as pd
from scipy.optimize import linprog

# or other ML libraries like sklearn, tensorflow, etc.

# Load the trained model

with open("nearest_neighbors_model.pkl", "rb") as f:
    nearest_neighbors_model = joblib.load(f)

with open("random_forest_regressor.pkl", "rb") as f:
    random_forest_model = joblib.load(f)

with open("nearest_neighbors_model.pkl", "wb") as f:
    pickle.dump(nearest_neighbors_model, f)

with open("random_forest_regressor.pkl", "wb") as f:
    pickle.dump(random_forest_model, f)

with open("nearest_neighbors_model.pkl", "rb") as f:
    random_forest_model = pickle.load(f)

with open("random_forest_regressor.pkl", "rb") as f:
    random_forest_model = pickle.load(f)

file_path = 'C:\\Chinmay\\Machine_Learning\\chain\\Min_macros_for_height_and_weight.xlsx'
macros_data = pd.read_excel(file_path)

# Load the previously fitted scaler
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# Now you can use transform() safely
# input_scaled = scaler_X.transform(macros_data)

# input_features = macros_data[['Protein (grams/day)', 'Fat (grams/day)', 'Carbs (grams/day)']]

# Function for predicting nutritional recommendations
def predict_nutrition(height, weight, calories):
    # Scale the input
    input_data = np.array([[height, weight, calories]])
    input_scaled = scaler_X.transform(input_data)
    
    # Predict using the trained model
    prediction_scaled = random_forest_model.predict(input_scaled)
    
    # Inverse scale the output
    prediction = scaler_y.inverse_transform(prediction_scaled)
    return {
        'Protein (grams/day)': prediction[0][0],
        'Carbs (grams/day)': prediction[0][1],
        'Fat (grams/day)': prediction[0][2],
        'Sugar (grams/day)': prediction[0][3],
    }

file_path = 'C:\\Chinmay\\Machine_Learning\\chain\\Food_data_generated_with_dietIDs.xlsx'
data = pd.read_excel(file_path)

import numpy as np
import random

import numpy as np
import pandas as pd
import itertools
import random

import pandas as pd
import numpy as np
from itertools import combinations

import itertools
import numpy as np
import pandas as pd

def recommend_diet_plan(protein_target, fat_target, carbs_target, data, top_n=3):
    # Extract relevant columns
    food_items = data[['Food_name', 'Protein(g)', 'Total lipid (fat)(g)', 'Carbohydrate, by difference(g)']].dropna()

    # Rename columns for ease
    food_items.columns = ["food_name", "protein", "fat", "carbs"]

    # Convert to numpy array for faster computation
    food_array = food_items[["protein", "fat", "carbs"]].values
    food_names = food_items["food_name"].values

    best_combinations = []

    for _ in range(top_n):  # Generate multiple different meal plans
        selected_indices = []
        current_protein, current_fat, current_carbs = 0, 0, 0
        meal_plan = []

        while (current_protein < protein_target or current_fat < fat_target or current_carbs < carbs_target) and len(meal_plan) < 5:
            idx = random.randint(0, len(food_array) - 1)

            # Prevent selecting the same food repeatedly
            if idx in selected_indices:
                continue

            selected_indices.append(idx)
            meal_plan.append({
                "food_name": food_names[idx],
                "protein": food_array[idx][0],
                "fat": food_array[idx][1],
                "carbs": food_array[idx][2]
            })

            # Update macro totals
            current_protein += food_array[idx][0]
            current_fat += food_array[idx][1]
            current_carbs += food_array[idx][2]

            # Stop if we reach the macro targets or 5 items
            if (current_protein >= protein_target and current_fat >= fat_target and current_carbs >= carbs_target) or len(meal_plan) == 5:
                best_combinations.append({
                    "meals": meal_plan,
                    "total_protein": current_protein,
                    "total_fat": current_fat,
                    "total_carbs": current_carbs,
                    "recommendation": "Balanced meal based on your macro needs!"
                })
                break  # Move to the next diet plan

    return best_combinations