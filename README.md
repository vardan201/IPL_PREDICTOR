🏏 IPL Win Predictor
A machine learning-based web application that predicts the probability of a team winning an IPL match in real time, based on match conditions like target score, overs completed, and current score.

🚀 Project Overview
This project analyzes IPL match data and uses a logistic regression model to predict the probability of a team winning during the second innings of a match. The predictor is deployed using Streamlit for a simple and interactive UI.

📂 Dataset
Two datasets were used for model training:

matches.csv: Contains match-level information.

deliveries.csv: Ball-by-ball level data of each match.

These datasets were preprocessed to create a structured dataset for model training.

🛠️ Features
Predicts win probability of batting team during live matches.

Calculates:

Runs Left

Balls Left

Current Run Rate (CRR)

Required Run Rate (RRR)

Interactive UI built with Streamlit.

Supports all major IPL teams and match cities.

🧠 Machine Learning
Model Used: Logistic Regression

Preprocessing:

One-Hot Encoding for categorical features (batting_team, bowling_team, city)

Feature engineering: runs_left, balls_left, crr, rrr

Libraries: scikit-learn, pandas, numpy, streamlit, pickle

📊 Model Evaluation
Model trained on a cleaned and engineered dataset.

Train-test split: 80-20

Evaluation metric: Accuracy Score

Pipeline saved using pickle.
🖥️ How to Run
🔧 Prerequisites
Make sure you have the following installed:

bash
Copy
Edit
pip install pandas numpy scikit-learn streamlit
🏃‍♂️ Running the App
bash
Copy
Edit
streamlit run app.py
Replace app.py with the name of your deployment script if different.
🗂️ Project Structure
bash
Copy
Edit
├── app.py               # Streamlit app
├── pipe.pkl             # Trained ML model pipeline
├── matches.csv          # Match-level dataset
├── deliveries.csv       # Ball-by-ball dataset
├── IPL Win Predictor.ipynb  # Model training and EDA notebook
└── README.md            # Project documentation
