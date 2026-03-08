import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- Page Config ---
st.set_page_config(page_title="AI Prediction Engine", layout="wide")
st.title("🎯 Big/Small Prediction Pipeline")

# --- Step 1: Check Files ---
file_list = ["01-15 2.0.csv", "1-15.csv"]
missing_files = [f for f in file_list if not os.path.exists(f)]

if missing_files:
    st.error(f"❌ Missing files in GitHub: {', '.join(missing_files)}")
    st.stop()

# --- Step 2: Define the Engine ---
def run_full_pipeline():
    # Progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()

    # 1. Clean Data
    status_text.text("Cleaning Data...")
    dfs = []
    for f in file_list:
        temp_df = pd.read_csv(f)
        temp_df.columns = [c.strip() for c in temp_df.columns]
        target = [c for c in temp_df.columns if 'B/S' in c or 'S/B' in c][0]
        temp_df = temp_df.rename(columns={target: 'Outcome', 'Ser No': 'Serial'})
        temp_df['Outcome_Num'] = temp_df['Outcome'].map({'B': 1, 'S': 0})
        dfs.append(temp_df[['Serial', 'Outcome_Num']])
    
    data = pd.concat(dfs).sort_values('Serial').drop_duplicates().reset_index(drop=True)
    progress_bar.progress(25)

    # 2. Feature Engineering
    status_text.text("Engineering Features...")
    lags = 5
    for i in range(1, lags + 1):
        data[f'lag_{i}'] = data['Outcome_Num'].shift(i)
    data['rolling_avg'] = data['Outcome_Num'].rolling(window=3).mean().shift(1)
    data = data.dropna()
    
    features = [f'lag_{i}' for i in range(1, lags + 1)] + ['rolling_avg']
    X = data[features]
    y = data['Outcome_Num']
    progress_bar.progress(50)

    # 3. Train & Evaluate (Self Test / Backtest)
    status_text.text("Training Ensemble Models...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    gb = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)
    
    acc = accuracy_score(y_test, rf.predict(X_test))
    progress_bar.progress(75)

    # 4. Prediction Engine & Confidence
    status_text.text("Generating Final Prediction...")
    last_row = data.iloc[-1]
    next_feat = pd.DataFrame([{
        'lag_1': last_row['Outcome_Num'],
        'lag_2': last_row['lag_1'],
        'lag_3': last_row['lag_2'],
        'lag_4': last_row['lag_3'],
        'lag_5': last_row['lag_4'],
        'rolling_avg': (last_row['Outcome_Num'] + last_row['lag_1'] + last_row['lag_2']) / 3
    }])[features]

    prob = (rf.predict_proba(next_feat)[0][1] + gb.predict_proba(next_feat)[0][1]) / 2
    prediction = "BIG (B)" if prob > 0.5 else "SMALL (S)"
    confidence = prob if prob > 0.5 else (1 - prob)
    
    progress_bar.progress(100)
    status_text.text("Done!")

    # Display Results
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("PREDICTION", prediction)
    col2.metric("CONFIDENCE", f"{confidence*100:.2f}%")
    col3.metric("BACKTEST ACCURACY", f"{acc*100:.2f}%")

# --- UI Button ---
if st.button("START PREDICTION ENGINE"):
    run_full_pipeline()
