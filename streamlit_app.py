import streamlit as st
import pickle
import pandas as pd

# Load model pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Title
st.title('🏏 IPL Win Predictor')

# Teams and cities
teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# Layout for team selection
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the Bowling Team', sorted([team for team in teams if team != batting_team]))

# Host city
selected_city = st.selectbox('Select Host City', sorted(cities))

# Target and match progress
target = st.number_input('Target Score', min_value=1)

col3, col4 = st.columns(2)
with col3:
    score = st.number_input('Current Score', min_value=0, max_value=target-1)
with col4:
    overs = st.number_input('Overs Completed', min_value=0.0, max_value=20.0, step=0.1)

# Predict button
if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = int(120 - (overs * 6))

    if overs == 0 or balls_left <= 0:
        st.error("Overs must be > 0 and balls left must be > 0 for prediction.")
    else:
        crr = score / overs
        rrr = (runs_left * 6) / balls_left

        # Create input dataframe
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # Predict probability
        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        st.subheader("🧮 Win Probabilities")
        st.success(f"🏆 {batting_team} Win Probability: **{round(win * 100, 2)}%**")
        st.error(f"⚔️ {bowling_team} Win Probability: **{round(loss * 100, 2)}%**")

        st.progress(int(win * 100))
