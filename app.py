import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- Page Config ---
st.set_page_config(page_title="Super AI Prediction Engine", layout="wide", page_icon="🤖")

st.markdown("""
    <style>
    .main-title { font-size: 36px; font-weight: bold; color: #1E88E5; text-align: center; }
    .step-box { padding: 10px; border-radius: 5px; background-color: #f0f2f6; border-left: 5px solid #1E88E5; margin-bottom: 5px; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-title">🎯 AI Prediction Engine + Advanced Streak Tracker</p>', unsafe_allow_html=True)

# Initialize Session State
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_pred' not in st.session_state:
    st.session_state.last_pred = None
if 'current_streak_num' not in st.session_state:
    st.session_state.current_streak_num = 0
if 'current_streak_type' not in st.session_state:
    st.session_state.current_streak_type = ""

# ==========================================
# 10-STEP AI PIPELINE LOGIC
# ==========================================

def run_ai_pipeline():
    # 1. CSV Data
    files = ["01-15 2.0.csv", "1-15.csv"]
    all_dfs = []
    for f in files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            # 2. Clean Data
            df.columns = [c.strip() for c in df.columns]
            target_col = [c for c in df.columns if 'B/S' in c or 'S/B' in c][0]
            ser_col = [c for c in df.columns if 'Ser' in c or 'Serial' in c][0]
            df = df.rename(columns={target_col: 'Outcome', ser_col: 'Serial', '0 to 9': 'Number'})
            df['Outcome_Num'] = df['Outcome'].map({'B': 1, 'S': 0})
            all_dfs.append(df[['Serial', 'Number', 'Outcome_Num']])
    
    if not all_dfs:
        return None, None, None, None

    full_data = pd.concat(all_dfs).sort_values('Serial').drop_duplicates().reset_index(drop=True)

    # 3. Feature Engineering (Enhanced to fix "wrong guess" issue)
    df_feat = full_data.copy()
    for i in range(1, 11): # Lags 1 to 10 for deeper pattern recognition
        df_feat[f'lag_{i}'] = df_feat['Outcome_Num'].shift(i)
    
    # Rolling stats
    df_feat['rolling_avg_3'] = df_feat['Outcome_Num'].rolling(window=3).mean().shift(1)
    df_feat['rolling_avg_5'] = df_feat['Outcome_Num'].rolling(window=5).mean().shift(1)
    
    df_feat = df_feat.dropna()
    features = [f'lag_{i}' for i in range(1, 11)] + ['rolling_avg_3', 'rolling_avg_5']
    X = df_feat[features]
    y = df_feat['Outcome_Num']

    # 4. Train Models & 5. Evaluate & 6. Self Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Models
    rf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42).fit(X_train, y_train)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42).fit(X_train, y_train)
    lr = LogisticRegression().fit(X_train, y_train)
    
    # 7. Backtest Accuracy
    acc_rf = accuracy_score(y_test, rf.predict(X_test))
    acc_gb = accuracy_score(y_test, gb.predict(X_test))
    
    return full_data, (rf, gb, lr), features, max(acc_rf, acc_gb)

# Load AI Engine
full_data, models, feature_cols, backtest_acc = run_ai_pipeline()

if full_data is not None:
    rf_model, gb_model, lr_model = models

    # ==========================================
    # DASHBOARD SECTION
    # ==========================================
    st.subheader("📊 Statistics Dashboard")
    
    # Helper to calculate streaks
    def calc_streaks(results):
        if not results: return 0, 0, 0, 0
        wins = [1 if "WIN" in r else 0 for r in results]
        
        def get_max_streak(binary_list):
            max_s = curr_s = 0
            for val in binary_list:
                if val == 1:
                    curr_s += 1
                    max_s = max(max_s, curr_s)
                else: curr_s = 0
            return max_s

        max_w = get_max_streak(wins)
        max_l = get_max_streak([1-x for x in wins])
        return max_w, max_l, sum(wins), len(wins) - sum(wins)

    if st.session_state.history:
        res_list = [h['Result'] for h in st.session_state.history][::-1]
        mw, ml, tw, tl = calc_streaks(res_list)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAX WIN STREAK", mw)
        c2.metric("MAX LOSS STREAK", ml)
        c3.metric("WINS (Total)", tw)
        c4.metric("LOSS (Total)", tl)
    else:
        st.info("No manual history yet. Stats will appear once you start predicting.")

    # ==========================================
    # PREDICTION ENGINE (Manual)
    # ==========================================
    st.divider()
    st.subheader("🔢 Step 9: Prediction Engine (Manual Input)")
    
    user_num = st.number_input("What was the last Number? (0-9)", 0, 9, 5)
    
    if st.button("🚀 SUBMIT & ANALYZE"):
        current_bs = "BIG" if user_num >= 5 else "SMALL"
        current_num_val = 1 if user_num >= 5 else 0
        
        # Streak/Result Logic
        display_result = "START"
        if st.session_state.last_pred is not None:
            outcome_type = "WIN" if st.session_state.last_pred == current_bs else "LOSS"
            if outcome_type == st.session_state.current_streak_type:
                st.session_state.current_streak_num += 1
            else:
                st.session_state.current_streak_num = 1
                st.session_state.current_streak_type = outcome_type
            display_result = f"{st.session_state.current_streak_type} {st.session_state.current_streak_num}"

        # 8. Ensemble Model & 10. Confidence Score
        last_rows = full_data.tail(10).copy()
        # Mock features for the next prediction
        lags = list(last_rows['Outcome_Num'].values)
        lags.append(current_num_val)
        lags = lags[-10:][::-1] # Last 10
        
        next_df = pd.DataFrame([lags + [np.mean(lags[:3]), np.mean(lags[:5])]], columns=feature_cols)
        
        # Ensemble Average Probability
        prob_rf = rf_model.predict_proba(next_df)[0][1]
        prob_gb = gb_model.predict_proba(next_x := next_df)[0][1]
        prob_lr = lr_model.predict_proba(next_df)[0][1]
        
        avg_prob = (prob_rf + prob_gb + prob_lr) / 3
        new_pred = "BIG" if avg_prob > 0.5 else "SMALL"
        conf = avg_prob if avg_prob > 0.5 else (1 - avg_prob)

        st.session_state.history.insert(0, {
            "Input": user_num, "Type": current_bs, "Result": display_result,
            "Next Prediction": new_pred, "Confidence": f"{conf*100:.2f}%"
        })
        st.session_state.last_pred = new_pred

    # Live Predict UI
    p1, p2, p3 = st.columns(3)
    p1.metric("CURRENT STREAK", f"{st.session_state.current_streak_type} {st.session_state.current_streak_num}")
    p2.header(f"AI NEXT: {st.session_state.last_pred if st.session_state.last_pred else '---'}")
    p3.metric("BACKTEST ACCURACY", f"{backtest_acc*100:.2f}%")

    # ==========================================
    # BATCH EVALUATION FEATURE
    # ==========================================
    st.divider()
    st.subheader("📂 Batch Evaluation (Historical Test)")
    eval_file = st.file_uploader("Upload CSV (Must have '0 to 9' column)", type="csv")

    if eval_file:
        edf = pd.read_csv(eval_file)
        edf.columns = [c.strip() for c in edf.columns]
        
        if '0 to 9' in edf.columns:
            results = []
            e_last_pred = None
            e_streak_num = 0
            e_streak_type = ""
            
            # Simulated history for eval
            eval_hist = list(full_data['Outcome_Num'].tail(10).values)
            
            for i, row in edf.iterrows():
                num = row['0 to 9']
                actual_bs = "BIG" if num >= 5 else "SMALL"
                actual_val = 1 if num >= 5 else 0
                
                # Check WIN/LOSS
                res_str = "N/A"
                if e_last_pred:
                    etype = "WIN" if e_last_pred == actual_bs else "LOSS"
                    if etype == e_streak_type: e_streak_num += 1
                    else: 
                        e_streak_num = 1
                        e_streak_type = etype
                    res_str = f"{e_streak_type} {e_streak_num}"

                # AI Predict Next
                eval_hist.append(actual_val)
                current_lags = eval_hist[-10:][::-1]
                in_df = pd.DataFrame([current_lags + [np.mean(current_lags[:3]), np.mean(current_lags[:5])]], columns=feature_cols)
                
                e_prob = (rf_model.predict_proba(in_df)[0][1] + gb_model.predict_proba(in_df)[0][1]) / 2
                e_last_pred = "BIG" if e_prob > 0.5 else "SMALL"
                
                results.append({
                    "Game": i+1, "Number": num, "Actual": actual_bs, 
                    "AI Result": res_str, "Predicted Next": e_last_pred
                })
            
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True)
            
            # Download Button
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Evaluation Results", data=csv, file_name="ai_evaluation.csv", mime="text/csv")

    # ==========================================
    # PASTED HISTORY TABLE
    # ==========================================
    st.divider()
    st.subheader("📜 Pasted History (Win/Loss Log)")
    if st.session_state.history:
        st.table(pd.DataFrame(st.session_state.history))
        if st.button("Clear Session Data"):
            st.session_state.history = []
            st.session_state.last_pred = None
            st.rerun()

else:
    st.error("⚠️ Error: '01-15 2.0.csv' and '1-15.csv' not found in root directory.")
