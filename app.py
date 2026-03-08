import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==========================================
# 1. PROFESSIONAL UI & COLOR CONFIG
# ==========================================
st.set_page_config(page_title="Super AI Prediction Pro", layout="wide", page_icon="🔮")

st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #0e1117; color: #ffffff; }
    
    /* High Contrast Titles */
    .main-title { font-size: 45px; font-weight: 800; text-align: center; color: #00d4ff; text-shadow: 2px 2px 4px #000000; }
    
    /* Big/Small Colors */
    .big-box { padding: 20px; border-radius: 10px; background-color: #ff3131; color: white; font-weight: bold; text-align: center; font-size: 30px; }
    .small-box { padding: 20px; border-radius: 10px; background-color: #39ff14; color: black; font-weight: bold; text-align: center; font-size: 30px; }
    
    /* Result Styling */
    .win-text { color: #39ff14; font-weight: bold; }
    .loss-text { color: #ff3131; font-weight: bold; }
    
    /* Tables */
    .styled-table { width: 100%; border-collapse: collapse; background-color: #1f2937; border-radius: 8px; overflow: hidden; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. FAST AI PIPELINE (CACHE ENABLED)
# ==========================================

@st.cache_resource
def train_models(files):
    all_dfs = []
    for f in files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            df.columns = [c.strip() for c in df.columns]
            target_col = [c for c in df.columns if 'B/S' in c or 'S/B' in c][0]
            ser_col = [c for c in df.columns if 'Ser' in c or 'Serial' in c][0]
            df = df.rename(columns={target_col: 'Outcome', ser_col: 'Serial', '0 to 9': 'Number'})
            df['Outcome_Num'] = df['Outcome'].map({'B': 1, 'S': 0})
            all_dfs.append(df[['Serial', 'Number', 'Outcome_Num']])
    
    full_data = pd.concat(all_dfs).sort_values('Serial').drop_duplicates().reset_index(drop=True)

    # Feature Engineering
    df_feat = full_data.copy()
    for i in range(1, 11):
        df_feat[f'lag_{i}'] = df_feat['Outcome_Num'].shift(i)
    df_feat['rolling_avg_3'] = df_feat['Outcome_Num'].rolling(window=3).mean().shift(1)
    df_feat['rolling_avg_5'] = df_feat['Outcome_Num'].rolling(window=5).mean().shift(1)
    df_feat = df_feat.dropna()
    
    features = [f'lag_{i}' for i in range(1, 11)] + ['rolling_avg_3', 'rolling_avg_5']
    X = df_feat[features]
    y = df_feat['Outcome_Num']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))
    
    return full_data, (rf, gb), features, acc

# Load/Train once
data_files = ["01-15 2.0.csv", "1-15.csv"]
full_data, models, feature_cols, backtest_acc = train_models(data_files)

# Session States
if 'history' not in st.session_state: st.session_state.history = []
if 'last_pred' not in st.session_state: st.session_state.last_pred = None
if 'streak_n' not in st.session_state: st.session_state.streak_n = 0
if 'streak_t' not in st.session_state: st.session_state.streak_t = ""
if 'session_lags' not in st.session_state: 
    st.session_state.session_lags = list(full_data['Outcome_Num'].tail(10).values)

st.markdown('<p class="main-title">🔮 SUPER AI PREDICTION ENGINE</p>', unsafe_allow_html=True)

# ==========================================
# 3. DASHBOARD STATS
# ==========================================
if st.session_state.history:
    wins = [1 if "WIN" in h['Result'] else 0 for h in st.session_state.history]
    total_w = sum(wins)
    total_l = len(wins) - total_w
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("WINS", total_w)
    c2.metric("LOSSES", total_l)
    c3.metric("BACKTEST ACC", f"{backtest_acc*100:.1f}%")
    c4.metric("LIVE STREAK", f"{st.session_state.streak_t} {st.session_state.streak_n}")

# ==========================================
# 4. PREDICTION ENGINE (STRICT PATTERN)
# ==========================================
st.divider()
user_num = st.number_input("Enter Last Resulting Number (0-9):", 0, 9, 5)

if st.button("🚀 GET NEXT AI GUESS"):
    actual_t = "BIG" if user_num >= 5 else "SMALL"
    actual_v = 1 if user_num >= 5 else 0
    
    # 1. Sync Logic (Add current number to session memory to fix your error)
    st.session_state.session_lags.append(actual_v)
    current_context = st.session_state.session_lags[-11:-1][::-1] # Last 10 before current
    
    # 2. Result Logic
    res_str = "START"
    if st.session_state.last_pred:
        res_t = "WIN" if st.session_state.last_pred == actual_t else "LOSS"
        if res_t == st.session_state.streak_t: st.session_state.streak_n += 1
        else: 
            st.session_state.streak_n = 1
            st.session_state.streak_t = res_t
        res_str = f"{st.session_state.streak_t} {st.session_state.streak_n}"

    # 3. AI Predict Next
    # Use the session_lags to ensure pattern stays consistent with Batch Eval
    input_lags = st.session_state.session_lags[-10:][::-1]
    in_df = pd.DataFrame([input_lags + [np.mean(input_lags[:3]), np.mean(input_lags[:5])]], columns=feature_cols)
    
    prob = (models[0].predict_proba(in_df)[0][1] + models[1].predict_proba(in_df)[0][1]) / 2
    next_p = "BIG" if prob > 0.5 else "SMALL"
    conf = prob if prob > 0.5 else (1 - prob)

    st.session_state.history.insert(0, {
        "Input": user_num, "Type": actual_t, "Result": res_str,
        "Next Prediction": next_p, "Confidence": f"{conf*100:.1f}%"
    })
    st.session_state.last_pred = next_p

# Visual Prediction
if st.session_state.last_pred:
    color_class = "big-box" if st.session_state.last_pred == "BIG" else "small-box"
    st.markdown(f'<div class class="{color_class}">NEXT PREDICTION: {st.session_state.last_pred}</div>', unsafe_allow_html=True)

# ==========================================
# 5. HISTORY & BATCH EVAL
# ==========================================
tab1, tab2 = st.tabs(["📜 Live History", "📂 Batch Evaluation"])

with tab1:
    if st.session_state.history:
        st.table(pd.DataFrame(st.session_state.history))

with tab2:
    eval_file = st.file_uploader("Upload Batch File", type="csv")
    if eval_file:
        edf = pd.read_csv(eval_file)
        # Logic matches manual exactly now
        eval_lags = list(full_data['Outcome_Num'].tail(10).values)
        results = []
        el_pred = None
        es_n = 0; es_t = ""
        for i, row in edf.iterrows():
            n = row['0 to 9']; at = "BIG" if n >= 5 else "SMALL"; av = 1 if n >= 5 else 0
            estr = "N/A"
            if el_pred:
                et = "WIN" if el_pred == at else "LOSS"
                if et == es_t: es_n += 1
                else: es_n = 1; es_t = et
                estr = f"{es_t} {es_n}"
            eval_lags.append(av)
            lags = eval_lags[-10:][::-1]
            in_df = pd.DataFrame([lags + [np.mean(lags[:3]), np.mean(lags[:5])]], columns=feature_cols)
            ep = (models[0].predict_proba(in_df)[0][1] + models[1].predict_proba(in_df)[0][1]) / 2
            el_pred = "BIG" if ep > 0.5 else "SMALL"
            results.append({"Number": n, "Type": at, "Result": estr, "Next Pred": el_pred})
        st.dataframe(pd.DataFrame(results))
