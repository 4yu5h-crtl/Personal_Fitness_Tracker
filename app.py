import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import time
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Fitness Tracker",
    page_icon="💪",
    layout="wide"
)
st.markdown("# My Personalized Fitness Dashboard")
st.markdown("Welcome! This dashboard provides a tailored estimation of the calories you burn during your workouts. Simply enter your fitness details in the sidebar, and discover your personalized insights.")

st.sidebar.header("Enter Your Fitness Details:")

def user_input_features():
    st.sidebar.markdown("### Personal Information")
    age = st.sidebar.slider("👤 Age", 10, 100, 30)
    gender_button = st.sidebar.radio("⚧ Gender", ("Male", "Female"))
    
    st.sidebar.markdown("### Body Metrics")
    bmi = st.sidebar.slider("📊 BMI", 15, 40, 20)
    body_temp = st.sidebar.slider("🌡️ Body Temperature (°C)", 36, 42, 38)
    
    st.sidebar.markdown("### Exercise Data")
    duration = st.sidebar.slider("⏱️ Duration (minutes)", 0, 35, 15)
    heart_rate = st.sidebar.slider("❤️ Heart Rate (bpm)", 60, 130, 80)

    gender = 1 if gender_button == "Male" else 0

    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender
    }

    features = pd.DataFrame(data_model, index=[0])
    return features

df = user_input_features()

st.markdown("---")
st.header("Your Submitted Information")
st.text("Processing your data...")
progress_bar = st.progress(0)
for i in range(100):
    progress_bar.progress(i + 1)
    time.sleep(0.01)
st.write(df)

calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

exercise_df = exercise.merge(calories, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)

exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

for data in [exercise_train_data, exercise_test_data]:
    data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
    data["BMI"] = round(data["BMI"], 2)

exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]

X_test = exercise_test_data.drop("Calories", axis=1)
y_test = exercise_test_data["Calories"]

random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
random_reg.fit(X_train, y_train)

user_data = df.reindex(columns=X_train.columns, fill_value=0)

calorie_prediction = random_reg.predict(user_data)

st.markdown("---")
st.header("Calorie Burn Estimate")
st.text("Calculating your estimated calorie burn...")
prediction_progress = st.progress(0)
for i in range(100):
    prediction_progress.progress(i + 1)
    time.sleep(0.01)
st.write(f"{round(calorie_prediction[0], 2)} **kilocalories**")

st.markdown("---")
st.header("Comparable Workouts")
st.text("Retrieving similar workout sessions...")
similar_progress = st.progress(0)
for i in range(100):
    similar_progress.progress(i + 1)
    time.sleep(0.01)
calorie_range = [calorie_prediction[0] - 10, calorie_prediction[0] + 10]
similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]
st.write(similar_data.sample(5))

st.markdown("---")
st.header("Comparative Insights")
bool_age = (exercise_df["Age"] < user_data["Age"].values[0]).tolist()
bool_duration = (exercise_df["Duration"] < user_data["Duration"].values[0]).tolist()
bool_body_temp = (exercise_df["Body_Temp"] < user_data["Body_Temp"].values[0]).tolist()
bool_heart_rate = (exercise_df["Heart_Rate"] < user_data["Heart_Rate"].values[0]).tolist()

st.write("Your age exceeds that of", round(sum(bool_age) / len(bool_age), 2) * 100, "% of the dataset.")
st.write("Your workout duration is longer than", round(sum(bool_duration) / len(bool_duration), 2) * 100, "% of the records.")
st.write("Your heart rate is higher than", round(sum(bool_heart_rate) / len(bool_heart_rate), 2) * 100, "% of the users during exercise.")
st.write("Your body temperature during exercise is higher than", round(sum(bool_body_temp) / len(bool_body_temp), 2) * 100, "% of the users.")