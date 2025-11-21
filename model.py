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

