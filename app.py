import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import io

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

    # --- NEW: DASHBOARD SECTION ---
    st.divider()
    st.subheader("📊 Statistics Dashboard")
    
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        
        # Calculate Stats
        total_wins = len(hist_df[hist_df['Result'].str.contains("WIN", na=False)])
        total_loss = len(hist_df[hist_df['Result'].str.contains("LOSS", na=False)])
        
        # Streak Parsing
        win_streaks = [int(val.split()[1]) for val in hist_df['Result'] if "WIN" in val]
        loss_streaks = [int(val.split()[1]) for val in hist_df['Result'] if "LOSS" in val]
        
        max_win = max(win_streaks) if win_streaks else 0
        max_loss = max(loss_streaks) if loss_streaks else 0

        d_c1, d_c2, d_c3, d_c4 = st.columns(4)
        d_c1.metric("MAX WIN STREAK", max_win)
        d_c2.metric("MAX LOSS STREAK", max_loss)
        d_c3.metric("TOTAL WINS", total_wins)
        d_c4.metric("TOTAL LOSSES", total_loss)
    else:
        st.info("Start predicting to see dashboard statistics.")

    # --- INPUT SECTION ---
    st.divider()
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
            'lag_4': last_val['lag_2'], 'lag_5': last_val['Outcome_Num'],
            'rolling_avg_3': (current_num_val + last_val['Outcome_Num'] + last_val['lag_1']) / 3
        }])[features]

        prob = (rf.predict_proba(next_x)[0][1] + gb.predict_proba(next_x)[0][1]) / 2
        new_prediction = "BIG" if prob > 0.5 else "SMALL"
        confidence = prob if prob > 0.5 else (1 - prob)

        st.session_state.history.insert(0, {
            "Input": user_num,
            "Type": current_bs,
            "Result": display_result,
            "Next Pred": new_prediction,
            "Conf": f"{confidence*100:.2f}%"
        })
        st.session_state.last_pred = new_prediction

    # --- DISPLAY METRICS ---
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

    # --- NEW: EVALUATION FEATURE ---
    st.divider()
    st.subheader("📂 Batch Evaluation (Upload Results CSV)")
    eval_file = st.file_uploader("Upload CSV for Evaluation (Must have '0 to 9' column)", type="csv")
    
    if eval_file:
        eval_df = pd.read_csv(eval_file)
        eval_df.columns = [c.strip() for c in eval_df.columns]
        
        if '0 to 9' in eval_df.columns:
            eval_results = []
            e_last_pred = None
            e_streak_num = 0
            e_streak_type = ""
            
            # Start evaluation based on the last row of training data
            current_context = data.iloc[-1].copy()
            
            for index, row in eval_df.iterrows():
                u_num = row['0 to 9']
                actual_bs = "BIG" if u_num >= 5 else "SMALL"
                actual_num = 1 if u_num >= 5 else 0
                
                # Compare with previous prediction
                e_res_text = "START"
                if e_last_pred is not None:
                    is_win = (e_last_pred == actual_bs)
                    o_type = "WIN" if is_win else "LOSS"
                    if o_type == e_streak_type:
                        e_streak_num += 1
                    else:
                        e_streak_num = 1
                        e_streak_type = o_type
                    e_res_text = f"{e_streak_type} {e_streak_num}"
                
                # Predict Next
                e_next_x = pd.DataFrame([{
                    'lag_1': actual_num, 'lag_2': current_context['Outcome_Num'], 'lag_3': current_context['lag_1'],
                    'lag_4': current_context['lag_2'], 'lag_5': current_context['lag_3'],
                    'rolling_avg_3': (actual_num + current_context['Outcome_Num'] + current_context['lag_1']) / 3
                }])[features]
                
                e_prob = (rf.predict_proba(e_next_x)[0][1] + gb.predict_proba(e_next_x)[0][1]) / 2
                e_last_pred = "BIG" if e_prob > 0.5 else "SMALL"
                
                eval_results.append({
                    "Game No": index + 1,
                    "Actual Number": u_num,
                    "Actual Type": actual_bs,
                    "Evaluation Result": e_res_text,
                    "Next Prediction": e_last_pred
                })
                
                # Shift context for next row in file
                current_context['lag_3'] = current_context['lag_2']
                current_context['lag_2'] = current_context['lag_1']
                current_context['lag_1'] = current_context['Outcome_Num']
                current_context['Outcome_Num'] = actual_num

            res_df = pd.DataFrame(eval_results)
            st.dataframe(res_df, use_container_width=True)
            
            # CSV Download
            csv_buffer = io.StringIO()
            res_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Evaluation Result CSV",
                data=csv_buffer.getvalue(),
                file_name="evaluation_results.csv",
                mime="text/csv"
            )
        else:
            st.error("CSV must contain a '0 to 9' column.")

    # --- WIN/LOSS LOG ---
    st.divider()
    st.subheader("📜 Pasted History (Win/Loss Log)")
    if st.session_state.history:
        st.table(pd.DataFrame(st.session_state.history))
        if st.button("Reset Everything"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
else:
    st.error("No CSV files found.")
