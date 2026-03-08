import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# --- Page Config ---
st.set_page_config(page_title="AI Prediction Pro", layout="wide")
st.title("🎯 AI Prediction Engine + Win/Loss Tracker")

# Initialize Session State for History and Streaks
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_pred' not in st.session_state:
    st.session_state.last_pred = None
if 'streak' not in st.session_state:
    st.session_state.streak = 0
if 'streak_type' not in st.session_state:
    st.session_state.streak_type = "" # "Win" or "Loss"

# --- STEP 1 & 2: DATA LOADING ---
@st.cache_data
def get_clean_data():
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

data = get_clean_data()

if data is not None:
    # --- STEP 3: FEATURE ENGINEERING ---
    for i in range(1, 6):
        data[f'lag_{i}'] = data['Outcome_Num'].shift(i)
    data['rolling_avg_3'] = data['Outcome_Num'].rolling(window=3).mean().shift(1)
    train_df = data.dropna()
    features = [f'lag_{i}' for i in range(1, 6)] + ['rolling_avg_3']
    
    # --- STEP 4-7: MODEL TRAINING ---
    X = train_df[features]
    y = train_df['Outcome_Num']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    gb = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)
    model_acc = accuracy_score(y_test, rf.predict(X_test))

    # --- UI: INPUT SECTION ---
    st.subheader("🔢 Step 8: Enter Latest Number")
    user_num = st.number_input("What number just appeared?", 0, 9, 5)
    
    if st.button("SUBMIT & PREDICT NEXT"):
        current_bs = "BIG" if user_num >= 5 else "SMALL"
        current_num = 1 if user_num >= 5 else 0
        
        # --- WIN / LOSS LOGIC ---
        result_text = "N/A"
        if st.session_state.last_pred is not None:
            if st.session_state.last_pred == current_bs:
                result_text = "✅ WIN"
                if st.session_state.streak_type == "Win":
                    st.session_state.streak += 1
                else:
                    st.session_state.streak = 1
                    st.session_state.streak_type = "Win"
            else:
                result_text = "❌ LOSS"
                if st.session_state.streak_type == "Loss":
                    st.session_state.streak += 1
                else:
                    st.session_state.streak = 1
                    st.session_state.streak_type = "Loss"

        # --- STEP 9: PREDICTION ENGINE ---
        last_val = data.iloc[-1]
        next_x = pd.DataFrame([{
            'lag_1': current_num, 'lag_2': last_val['Outcome_Num'], 'lag_3': last_val['lag_1'],
            'lag_4': last_val['lag_2'], 'lag_5': last_val['lag_3'],
            'rolling_avg_3': (current_num + last_val['Outcome_Num'] + last_val['lag_1']) / 3
        }])[features]

        prob = (rf.predict_proba(next_x)[0][1] + gb.predict_proba(next_x)[0][1]) / 2
        new_prediction = "BIG" if prob > 0.5 else "SMALL"
        confidence = prob if prob > 0.5 else (1 - prob)

        # Update History
        st.session_state.history.insert(0, {
            "Input Num": user_num,
            "Type": current_bs,
            "Result": result_text,
            "Next Prediction": new_prediction,
            "Confidence": f"{confidence*100:.2f}%"
        })
        
        # Store prediction for next turn
        st.session_state.last_pred = new_prediction

    # --- STEP 10: DISPLAY RESULTS ---
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("CURRENT STREAK", f"{st.session_state.streak} {st.session_state.streak_type}")
    with col2:
        if st.session_state.last_pred:
            st.header(f"NEXT: {st.session_state.last_pred}")
        else:
            st.header("Enter Number to Start")
    with col3:
        st.metric("BACKTEST ACCURACY", f"{model_acc*100:.2f}%")

    # --- HISTORY TABLE ---
    st.subheader("📜 Pasted History (Win/Loss Log)")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.table(history_df)
        
        if st.button("Clear History"):
            st.session_state.history = []
            st.session_state.streak = 0
            st.session_state.last_pred = None
            st.rerun()

else:
    st.error("Missing CSV files. Please upload them to GitHub.")
