import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Page Config
st.set_page_config(page_title="Big/Small AI", layout="wide")
st.title("🎯 AI Prediction Engine (v2.0)")

# Step 1 & 2: Load and Clean
@st.cache_data
def get_data():
    files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if not files: return None
    
    all_dfs = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        target = [c for c in df.columns if 'B/S' in c or 'S/B' in c][0]
        ser = [c for c in df.columns if 'Ser' in c or 'Serial' in c][0]
        df = df.rename(columns={target: 'Outcome', ser: 'Serial'})
        df['Outcome_Num'] = df['Outcome'].map({'B': 1, 'S': 0})
        all_dfs.append(df[['Serial', 'Outcome_Num']])
    
    return pd.concat(all_dfs).sort_values('Serial').drop_duplicates().dropna()

data = get_data()

if data is not None:
    # Step 3: Feature Engineering
    for i in range(1, 6):
        data[f'lag_{i}'] = data['Outcome_Num'].shift(i)
    data['rolling_avg_3'] = data['Outcome_Num'].rolling(window=3).mean().shift(1)
    train_df = data.dropna()
    
    features = [f'lag_{i}' for i in range(1, 6)] + ['rolling_avg_3']
    X = train_df[features]
    y = train_df['Outcome_Num']

    # Step 4-7: Models
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    gb = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)
    
    acc = accuracy_score(y_test, rf.predict(X_test))

    # Manual Input Section
    st.subheader("🔢 Manual Input")
    user_num = st.number_input("Enter the last number appeared (0-9):", 0, 9, 5)
    
    if st.button("GET NEXT PREDICTION"):
        current_outcome = 1 if user_num >= 5 else 0
        last_val = data.iloc[-1]
        
        # Step 8-10: Prediction Engine
        next_x = pd.DataFrame([{
            'lag_1': current_outcome,
            'lag_2': last_val['Outcome_Num'],
            'lag_3': last_val['lag_1'],
            'lag_4': last_val['lag_2'],
            'lag_5': last_val['lag_3'],
            'rolling_avg_3': (current_outcome + last_val['Outcome_Num'] + last_val['lag_1']) / 3
        }])[features]

        prob = (rf.predict_proba(next_x)[0][1] + gb.predict_proba(next_x)[0][1]) / 2
        prediction = "BIG (B)" if prob > 0.5 else "SMALL (S)"
        conf = prob if prob > 0.5 else (1 - prob)

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("PREDICTION", prediction)
        c2.metric("CONFIDENCE", f"{conf*100:.2f}%")
        c3.metric("WIN RATE", f"{acc*100:.2f}%")
else:
    st.error("Upload CSV files to GitHub.")
