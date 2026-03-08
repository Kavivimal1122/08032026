import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Page Configuration
st.set_page_config(page_title="AI Prediction Engine", layout="wide")
st.title("🎯 Real-Time Big/Small Prediction Engine")

# --- STEP 1 & 2: DATA LOADING & CLEANING ---
@st.cache_data
def load_historical_data():
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if not csv_files:
        return None

    all_dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df.columns = [c.strip() for c in df.columns]
            target_col = [c for c in df.columns if 'B/S' in c or 'S/B' in c][0]
            ser_col = [c for c in df.columns if 'Ser' in c or 'Serial' in c][0]
            
            df = df.rename(columns={target_col: 'Outcome', ser_col: 'Serial'})
            df['Outcome_Num'] = df['Outcome'].map({'B': 1, 'S': 0})
            all_dfs.append(df[['Serial', 'Outcome_Num']])
        except:
            continue
            
    if not all_dfs:
        return None
    return pd.concat(all_dfs).sort_values('Serial').drop_duplicates().dropna()

data = load_historical_data()

if data is not None:
    # --- STEP 3: FEATURE ENGINEERING ---
    # Create lag features for training
    lags = 5
    for i in range(1, lags + 1):
        data[f'lag_{i}'] = data['Outcome_Num'].shift(i)
    data['rolling_avg_3'] = data['Outcome_Num'].rolling(window=3).mean().shift(1)
    train_df = data.dropna()

    features = [f'lag_{i}' for i in range(1, lags + 1)] + ['rolling_avg_3']
    X = train_df[features]
    y = train_df['Outcome_Num']

    # --- STEP 4-7: MODEL TRAINING & EVALUATION ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    gb = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)
    lr = LogisticRegression().fit(X_train, y_train)
    
    acc = accuracy_score(y_test, rf.predict(X_test))

    # --- UI: MANUAL INPUT SECTION ---
    st.divider()
    st.subheader("🔢 Manual Input: What was the last number?")
    
    # User inputs the single number (0-9)
    user_num = st.number_input("Enter the latest number (0 to 9):", min_value=0, max_value=9, step=1)
    
    if st.button("PREDICT NEXT RESULT"):
        # Convert user number to Big (1) or Small (0)
        # Logic: 0-4 = Small, 5-9 = Big
        current_outcome = 1 if user_num >= 5 else 0
        current_label = "BIG" if current_outcome == 1 else "SMALL"
        
        st.write(f"Identified latest as: **{current_label}** (Number {user_num})")

        # --- STEP 8-10: ENSEMBLE PREDICTION ENGINE ---
        # Get historical context for the remaining lags from the CSV
        last_val = data.iloc[-1]
        
        # Build the feature vector for the NEXT period
        # Lag 1 is now the number you just typed in
        next_features = pd.DataFrame([{
            'lag_1': current_outcome,
            'lag_2': last_val['Outcome_Num'],
            'lag_3': last_val['lag_1'],
            'lag_4': last_val['lag_2'],
            'lag_5': last_val['lag_3'],
            'rolling_avg_3': (current_outcome + last_val['Outcome_Num'] + last_val['lag_1']) / 3
        }])[features]

        # Get probabilities from ensemble
        prob = (rf.predict_proba(next_features)[0][1] + 
                gb.predict_proba(next_features)[0][1] + 
                lr.predict_proba(next_features)[0][1]) / 3
        
        prediction = "BIG (B)" if prob > 0.5 else "SMALL (S)"
        confidence = prob if prob > 0.5 else (1 - prob)

        # Final Display
        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("PREDICTED NEXT", prediction)
        with c2:
            st.metric("CONFIDENCE SCORE", f"{confidence*100:.2f}%")
        with c3:
            st.metric("MODEL ACCURACY", f"{acc*100:.2f}%")
        
        if confidence > 0.60:
            st.success("🔥 High Confidence Prediction!")
        else:
            st.warning("⚠️ Low Confidence - Trend is uncertain.")

    st.divider()
    st.info(f"The engine is currently using {len(data)} rows of historical data for pattern recognition.")

else:
    st.error("Please upload the CSV files to your GitHub folder to start.")
