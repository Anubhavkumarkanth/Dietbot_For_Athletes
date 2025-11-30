import streamlit as st
import pandas as pd
import random
from datetime import datetime
from model import predict_nutrition, recommend_diet_plan

file_path = 'C:\\Chinmay\\Machine_Learning\\chain\\Food_data_generated_with_dietIDs.xlsx'
data = pd.read_excel(file_path)

# Set page configuration
st.set_page_config(
    page_title="Athlete Diet Recommendation Chatbot",
    page_icon="🥗",
    layout="wide"
)

# Sidebar for user inputs
st.sidebar.title("User Input")
st.sidebar.markdown("Provide your details to get personalized diet recommendations.")

# User inputs
height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, value=170)
weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
sport = st.sidebar.selectbox("Sport", [
    "Basketball", "Soccer", "Swimming", "Running", "Cycling", "Tennis",
    "Weightlifting", "Volleyball", "Gymnastics", "Boxing", "Wrestling",
    "Hockey", "Football", "Badminton", "Table Tennis", "Track and Field",
    "Baseball", "Softball", "Golf", "Skiing", "Rowing", "Martial Arts",
    "Yoga", "CrossFit"
])
duration = st.sidebar.slider("Duration of Activity (minutes)", min_value=10, max_value=180, value=60)
activity_level = st.sidebar.selectbox(
    "Activity Level",
    ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extra Active"]
)
goal = st.sidebar.selectbox("Fitness Goal", ["Muscle Gain", "Weight Loss", "Weight Gain"])

# MET values for sports
MET_VALUES = {
    "Basketball": 6.5,
    "Football": 9.0,
    "Swimming": 8.0,
    "Running": 9.8,
    "Cycling": 7.5,
    "Tennis": 7.3,
    "Weightlifting": 3.0,
    "Volleyball": 4.0,
    "Gymnastics": 3.8,
    "Boxing": 9.0,
    "Wrestling": 6.0,
    "Hockey": 8.5,
    "Rugby": 8.0,
    "Badminton": 5.5,
    "Table Tennis": 4.0,
    "Track and Field": 9.0,
    "Baseball": 5.0,
    "Softball": 5.0,
    "Golf": 4.8,
    "Skiing": 7.0,
    "Rowing": 6.0,
    "Martial Arts": 10.0,
    "Yoga": 2.5,
    "CrossFit": 9.5
}

def calculate_calories_burned(weight, duration, met):
    return met * weight * (duration / 60)

calories = calculate_calories_burned(weight, duration, MET_VALUES[sport])

# Main chatbot interface
st.title("🥗 Athlete Diet Recommendation Chatbot")
st.markdown("""
Welcome to your personalized diet recommendation assistant! 
Get tailored meal plans based on your activity level and fitness goals.
""")

# Display current date and time
current_time = datetime.now().strftime("%A, %B %d, %Y - %I:%M %p")
st.write(f"📅 Current Date and Time: {current_time}")

# Predict nutrition and recommend meal plans when user submits input
if st.sidebar.button("Get Recommendations"):
    nutrition_data = predict_nutrition(weight, height, calories)
    
    st.success(f"Calories Burned During {sport}: {calories} kcal")
    
    with st.expander("View Nutritional Breakdown"):
        st.write(f"Protein: {nutrition_data['Protein (grams/day)']}")
        st.write(f"Carbs: {nutrition_data['Carbs (grams/day)']} ")
        st.write(f"Fat: {nutrition_data['Fat (grams/day)']} ")

    recommendations = recommend_diet_plan(nutrition_data["Protein (grams/day)"], nutrition_data["Fat (grams/day)"], nutrition_data["Carbs (grams/day)"], data)

    # Display recommended meal plans
    st.subheader("🍽 Recommended Meal Plans")

    for i, plan in enumerate(recommendations):
        with st.expander(f"Meal Plan {i + 1}"):
            for meal in plan["meals"]:
                st.write(f"- {meal['food_name']} (Protein: {meal['protein']}g, Fat: {meal['fat']}g, Carbs: {meal['carbs']}g)")
            st.write(f"**Total Macros:** {plan['total_protein']}g Protein, {plan['total_fat']}g Fat, {plan['total_carbs']}g Carbs")
            st.write(f"💡 *Tip:* {plan['recommendation']}")