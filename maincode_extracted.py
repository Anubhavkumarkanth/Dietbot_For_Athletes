%pip install scikit-learn==1.6.1

import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import numpy as np

file_path = 'Min_macros_for_height_and_weight.xlsx' # Update this to the filename after uploading
data = pd.ExcelFile(file_path)

df = data.parse('Sheet1')

df.describe()

df.isnull()

input_features = ['Height', 'Weight', 'Food Energy (Calories/day)']
output_features = ['Protein (grams/day)', 'Carbs (grams/day)', 'Fat (grams/day)', 'Sugar (grams/day)']

df['Food Energy (Calories/day)'] = df['Food Energy (Calories/day)'].astype(str)  # Convert to string
df['Food Energy (Calories/day)'] = df['Food Energy (Calories/day)'].str.replace(',', '').astype(float)

missing_inputs = df[input_features].isnull().sum()
missing_outputs = df[output_features].isnull().sum()

print("Missing values in input features:\n", missing_inputs)
print("Missing values in output features:\n", missing_outputs)

# Drop rows with missing values (or you can impute them as needed)
df = df.dropna(subset=input_features + output_features)

missing_after_dropping = df.isnull().sum()
print("Missing values after dropping rows:\n", missing_after_dropping)

X = df[input_features]
y = df[output_features]

# Standardize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

input_data = df[input_features].values

input_df = pd.DataFrame(input_data, columns=input_features)

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Now transform using the fitted scaler
input_scaled = scaler_X.transform(input_df)

import joblib
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

def predict_nutrition(height, weight, calories):
    # Scale the input
    input_data = np.array([[height, weight, calories]])
    input_scaled = scaler_X.transform(input_data)

    # Predict using the trained model
    prediction_scaled = model.predict(input_scaled)

    # Inverse scale the output
    prediction = scaler_y.inverse_transform(prediction_scaled)
    return {
        'Protein (grams/day)': prediction[0][0],
        'Carbs (grams/day)': prediction[0][1],
        'Fat (grams/day)': prediction[0][2],
        'Sugar (grams/day)': prediction[0][3],
    }

example_input = {'Height': 182, 'Weight': 69, 'Calories': 3000}
result = predict_nutrition(example_input['Height'], example_input['Weight'], example_input['Calories'])
print("\nNutritional Recommendations for Input:", example_input)
print(result)

example_input = {'Height': 101, 'Weight': 43, 'Calories': 2000}
result = predict_nutrition(example_input['Height'], example_input['Weight'], example_input['Calories'])
print(result)

nutritional_recommendations = [
    result['Protein (grams/day)'],
    result['Carbs (grams/day)'],
    result['Fat (grams/day)'],
    result['Sugar (grams/day)']
]

# Display the result
print("\nNutritional Recommendations for Input:", example_input)
print(result)
print("\nNutritional Recommendations List:", nutritional_recommendations)

print(result)

import joblib

# Save the trained Random Forest Regressor model
rf_model_filename = "random_forest_regressor.pkl"
joblib.dump(model, rf_model_filename)

print(f"Model saved as {rf_model_filename}")

protein = result['Protein (grams/day)']
fat = result['Fat (grams/day)']
carbs = result['Carbs (grams/day)']

file_path = 'Food_data_generated_with_dietIDs.csv'
food_data = pd.read_csv(file_path, encoding='latin1')

food_data.head()

food_data.info()

# Extract relevant input features (protein, fat, carbs) and output columns (food names and nutritional content)
# Input features
input_features = food_data[['Protein(g)', 'Total lipid (fat)(g)', 'Carbohydrate, by difference(g)']]

# Extract food names (assuming the Food_name column is the correct one)
food_names = food_data['Food_name']

# Combine extracted columns for clarity and processing
selected_data = pd.concat([input_features, food_names], axis=1)

# Check for missing values and overall integrity
selected_data.info(), selected_data.head()

for column in ["proteing", "total_lipid_fatg", "carbohydrate,_by_differenceg", "waterg"]:
    if column in input_features.columns:
        mean_value = input_features[column][input_features[column] >= 0].mean()
        input_features[column] = input_features[column].apply(lambda x: mean_value if x < 0 else x)

# Special handling for "waterg" column
if "waterg" in input_features.columns:
    mean_value_waterg = input_features["waterg"].dropna().mean()
    input_features["waterg"] = input_features["waterg"].apply(lambda x: mean_value_waterg if pd.isna(x) or x < 0 else x)

X = input_features.to_numpy()

# Create a model using Nearest Neighbors for recommendation
nn_model = NearestNeighbors(n_neighbors=5, metric='euclidean')
nn_model.fit(X)

def recommend_foods(protein, fat, carbs, model, food_names_df, input_features_df):
    # Input as a vector
    input_vector = np.array([[protein, fat, carbs]])
    distances, indices = model.kneighbors(input_vector)
    recommendations = []
    for idx in indices[0]:
        # Correctly access the food name from the Series using the index
        food_name = food_names_df.iloc[idx]
        # Ensure the index exists in input_features_df before accessing
        if idx in input_features_df.index:
            nutrition_data = input_features_df.iloc[idx]
            recommendations.append({
                "food_name": food_name,
                "protein": nutrition_data["Protein(g)"], # Use correct column name
                "fat": nutrition_data["Total lipid (fat)(g)"], # Use correct column name
                "carbs": nutrition_data["Carbohydrate, by difference(g)"], # Use correct column name
                # Assuming 'waterg' exists and is a valid column in input_features_df if needed
                # If 'waterg' is not in input_features_df, you might need to get it from the original food_data
                "water": f"{random.uniform(2, 3):.2f}L"  # Include water if available
            })
    return pd.DataFrame(recommendations).to_dict()

sample_recommendations = recommend_foods(protein, fat, carbs, nn_model, food_names, input_features)
sample_recommendations

accuracy = accuracy_score(actual_food_indices, predicted_food_indices)
mse = mean_squared_error(actual_food_indices, predicted_food_indices)
r2 = r2_score(actual_food_indices, predicted_food_indices)

import joblib

# Save the trained NearestNeighbors model
model_filename = "nearest_neighbors_model.pkl"
joblib.dump(nn_model, model_filename)

print(f"Model saved as {model_filename}")

file_path = 'Min_macros_for_height_and_weight.xlsx'
bmi_data = pd.read_excel(file_path, sheet_name='Sheet1')

bmi_data.rename(columns=lambda x: x.strip(), inplace=True)  # Remove extra spaces in column names
if 'Food Energy (Calories/day)' in bmi_data.columns and bmi_data['Food Energy (Calories/day)'].dtype == 'object':
    bmi_data['Food Energy (Calories/day)'] = bmi_data['Food Energy (Calories/day)'].str.replace(',', '').astype(float)

if 'bmi_range' in bmi_data.columns:
    # Plot 1: BMI Distribution by Category
    bmi_counts = bmi_data['bmi_range'].value_counts()
    plt.figure(figsize=(8, 5))
    plt.bar(bmi_counts.index, bmi_counts.values, color='skyblue')
    plt.title('BMI Distribution by Category', fontsize=16)
    plt.xlabel('BMI Range', fontsize=12)
    plt.ylabel('Number of Individuals', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
else:
    print("Column 'bmi_range' not found in the dataset.")

categories = ['Protein (grams/day)', 'Carbs (grams/day)', 'Fat (grams/day)']
valid_categories = [col for col in categories if col in bmi_data.columns]
if valid_categories:
    mean_values = bmi_data[valid_categories].mean()

    plt.figure(figsize=(8, 5))
    plt.bar(mean_values.index, mean_values.values, color=['#ff9999', '#66b3ff', '#99ff99'])
    plt.title('Average Daily Nutritional Requirements', fontsize=16)
    plt.ylabel('Grams per Day', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
else:
    print("No valid nutritional requirement columns found in the dataset.")

import matplotlib.pyplot as plt
import numpy as np

# Assuming `sample_recommendations` is the DataFrame shown in the image
# Extract data
food_names = sample_recommendations['food_name']
protein = sample_recommendations['protein']
fat = sample_recommendations['fat']
carbs = sample_recommendations['carbs']

# Bar chart to visualize Protein, Fat, and Carbs content
x = np.arange(len(food_names))  # Food indices
width = 0.2  # Bar width

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width, protein, width, label='Protein', color='skyblue')
bars2 = ax.bar(x, fat, width, label='Fat', color='orange')
bars3 = ax.bar(x + width, carbs, width, label='Carbs', color='green')

# Adding labels and title
ax.set_xlabel('Food Names', fontsize=12)
ax.set_ylabel('Nutritional Content (grams)', fontsize=12)
ax.set_title('Nutritional Content of Recommended Foods', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(food_names, rotation=45, ha='right', fontsize=10)
ax.legend()

# Adding value labels on top of bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns

# Perform K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10) # Added n_init for newer sklearn versions
input_features_cleaned = input_features[['Protein(g)', 'Total lipid (fat)(g)', 'Carbohydrate, by difference(g)']].dropna()
kmeans.fit(input_features_cleaned)
clusters = kmeans.labels_

# Reduce dimensions for visualization using PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(input_features_cleaned)

# Plot clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=clusters, palette="Set1", s=60)
plt.title("Clustering of Foods Based on Nutritional Profiles", fontsize=14)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Pairplot for protein, fat, carbs
sns.pairplot(input_features[['Protein(g)', 'Total lipid (fat)(g)', 'Carbohydrate, by difference(g)']].dropna(),
             diag_kind="kde",
             kind="scatter",
             height=2.5)
plt.suptitle("Pairwise Relationships Across Nutritional Features", y=1.02, fontsize=16)
plt.show()

from math import pi

# Select one food for visualization
# Access data from the dictionary for the first recommended food (index 0)
food_name_to_plot = sample_recommendations['food_name'][0]
protein_to_plot = sample_recommendations['protein'][0]
fat_to_plot = sample_recommendations['fat'][0]
carbs_to_plot = sample_recommendations['carbs'][0]
water_to_plot = sample_recommendations['water'][0] # Assuming 'water' key exists and is relevant for plotting

# Radar chart setup
categories = ['Protein', 'Fat', 'Carbs', 'Water']
values = [protein_to_plot, fat_to_plot, carbs_to_plot]

# Convert water string to float for plotting if necessary, or handle as a separate visual element
# For this radar chart, let's exclude water as it's in a different unit and format.
# If you need to include water, you might need a different visualization or unit conversion.
categories = ['Protein', 'Fat', 'Carbs']
values = [protein_to_plot, fat_to_plot, carbs_to_plot]


values += values[:1]  # Close the radar chart
angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, values, color='skyblue', alpha=0.4)
ax.plot(angles, values, color='blue', linewidth=2)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_title(f"Nutritional Profile of {food_name_to_plot}", fontsize=14)
# Set limits for each axis to ensure consistent scaling across different food items
ax.set_ylim(0, max(values) * 1.2) # Adjust limit based on the max value to make plot readable
ax.grid(True)

plt.show()

from math import pi

# Select multiple foods for visualization
selected_data = input_features.sample(3)  # Randomly sample 3 foods
categories = ['Protein(g)', 'Total lipid (fat)(g)', 'Carbohydrate, by difference(g)']

# Create radar charts
fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))
for i, (idx, row) in enumerate(selected_data.iterrows()):
    values = row[categories].values.tolist()
    values += values[:1]  # Close the radar chart
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]

    ax = axes[i]
    ax.fill(angles, values, color='skyblue', alpha=0.4)
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(f"Food {idx} Nutritional Profile", fontsize=14)

plt.tight_layout()
plt.show()

# Load the dataset
file_path = 'Food_data_generated_with_dietIDs.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')

X = data[['Protein(g)', 'Total lipid (fat)(g)', 'Carbohydrate, by difference(g)']]
y = data['Food_name']

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

features = data[['Protein(g)', 'Total lipid (fat)(g)', 'Carbohydrate, by difference(g)']]

# Scale the features (important for nearest neighbor)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

model = NearestNeighbors(n_neighbors=5)
model.fit(scaled_features)

# Use the predicted macronutrient values from the 'result' dictionary as input
input_protein = result['Protein (grams/day)']
input_fat = result['Fat (grams/day)']
input_carbs = result['Carbs (grams/day)']

input_data = [[input_protein, input_fat, input_carbs]]  # Use the numerical values

# Scale the input data using the scaler fitted on the food features
input_scaled = scaler.transform(input_data)

distances, indices = model.kneighbors(input_scaled)

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using indices and distances score
    """
    # Print results
    print("Nearest Neighbors (Indexes):", indices)
    print("Distances to Neighbors:", distances)

evaluate_model(model, X_test, y_test)

import joblib

# Save the trained NearestNeighbors model
model_filename = "chatbot-model.pkl"
joblib.dump(model, model_filename)

print(f"Model saved as {model_filename}")

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

recommendations = recommend_diet_plan(result['Protein (grams/day)'], result['Fat (grams/day)'], result['Carbs (grams/day)'], data, top_n=3)
print(recommendations)

# Example macro target
protein_target = 69
fat_target = 50
carbs_target = 65

recommended_foods = recommend_diet_plan(protein_target, fat_target, carbs_target, data)
print(recommended_foods)

