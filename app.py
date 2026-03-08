import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# --- Page Config ---
st.set_page_config(page_title="AI Prediction Pro", layout="wide")
st.title("🎯 AI Prediction Engine + Streak Tracker")

# Initialize Session State
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_pred' not in st.session_state:
    st.session_state.last_pred = None
if 'current_streak_num' not in st.session_state:
    st.session_state.current_streak_num = 0
if 'current_streak_type' not in st.session_state:
    st.session_state.current_streak_type = ""

# --- DATA LOADING & CLEANING ---
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
    # --- FEATURE ENGINEERING & TRAINING ---
    for i in range(1, 6):
        data[f'lag_{i}'] = data['Outcome_Num'].shift(i)
    data['rolling_avg_3'] = data['Outcome_Num'].rolling(window=3).mean().shift(1)
    train_df = data.dropna()
    features = [f'lag_{i}' for i in range(1, 6)] + ['rolling_avg_3']
    
    X = train_df[features]
    y = train_df['Outcome_Num']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    gb = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)
    model_acc = accuracy_score(y_test, rf.predict(X_test))

    # --- INPUT SECTION ---
    st.subheader("🔢 Enter Latest Number")
    user_num = st.number_input("What number just appeared?", 0, 9, 5)
    
    if st.button("SUBMIT & PREDICT NEXT"):
        current_bs = "BIG" if user_num >= 5 else "SMALL"
        current_num_val = 1 if user_num >= 5 else 0
        
        # --- STREAK CALCULATION ---
        display_result = "START"
        if st.session_state.last_pred is not None:
            is_win = (st.session_state.last_pred == current_bs)
            outcome_type = "WIN" if is_win else "LOSS"
            
            if outcome_type == st.session_state.current_streak_type:
                st.session_state.current_streak_num += 1
            else:
                st.session_state.current_streak_num = 1
                st.session_state.current_streak_type = outcome_type
            
            display_result = f"{st.session_state.current_streak_type} {st.session_state.current_streak_num}"

        # --- PREDICTION ENGINE ---
        last_val = data.iloc[-1]
        next_x = pd.DataFrame([{
            'lag_1': current_num_val, 'lag_2': last_val['Outcome_Num'], 'lag_3': last_val['lag_1'],
            'lag_4': last_val['lag_2'], 'lag_5': last_val['lag_3'],
            'rolling_avg_3': (current_num_val + last_val['Outcome_Num'] + last_val['lag_1']) / 3
        }])[features]

        prob = (rf.predict_proba(next_x)[0][1] + gb.predict_proba(next_x)[0][1]) / 2
        new_prediction = "BIG" if prob > 0.5 else "SMALL"
        confidence = prob if prob > 0.5 else (1 - prob)

        # Update History Table
        st.session_state.history.insert(0, {
            "Input": user_num,
            "Type": current_bs,
            "Result": display_result,
            "Next Pred": new_prediction,
            "Conf": f"{confidence*100:.2f}%"
        })
        
        st.session_state.last_pred = new_prediction

    # --- DISPLAY ---
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("STREAK", f"{st.session_state.current_streak_type} {st.session_state.current_streak_num}")
    with c2:
        if st.session_state.last_pred:
            st.header(f"NEXT: {st.session_state.last_pred}")
        else:
            st.header("Ready...")
    with c3:
        st.metric("ACCURACY", f"{model_acc*100:.2f}%")

    # --- WIN/LOSS LOG ---
    st.subheader("📜 Pasted History (Win/Loss Log)")
    if st.session_state.history:
        st.table(pd.DataFrame(st.session_state.history))
        
        if st.button("Reset Everything"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
else:
    st.error("No CSV files found.")
