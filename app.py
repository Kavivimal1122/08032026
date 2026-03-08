import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==========================================
# 1. UI CONFIGURATION (PROFESSIONAL CSS)
# ==========================================
st.set_page_config(page_title="Super AI Prediction Pro", layout="wide", page_icon="🔮")

st.markdown("""
    <style>
    /* Main Background and Fonts */
    .main { background-color: #0e1117; color: #ffffff; }
    .stApp { background: linear-gradient(135deg, #111827 0%, #1f2937 100%); }
    
    /* Professional Titles */
    .main-title { 
        font-size: 45px; font-weight: 800; text-align: center; 
        background: -webkit-linear-gradient(#4facfe, #00f2fe);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 30px;
    }
    
    /* Result Cards */
    .prediction-card {
        padding: 30px; border-radius: 15px; text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3); margin-bottom: 20px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Color Mapping */
    .big-text { color: #FF3131; font-weight: bold; text-shadow: 0 0 10px rgba(255, 49, 49, 0.4); }
    .small-text { color: #39FF14; font-weight: bold; text-shadow: 0 0 10px rgba(57, 255, 20, 0.4); }
    
    /* Metric Styling */
    [data-testid="stMetricValue"] { font-size: 28px !important; font-weight: 700 !important; }
    
    /* Table Styling */
    .styled-table {
        width: 100%; border-collapse: collapse; margin: 25px 0; font-size: 18px;
        text-align: left; border-radius: 10px; overflow: hidden;
    }
    .styled-table thead tr { background-color: #009879; color: #ffffff; text-align: left; font-weight: bold; }
    .styled-table th, .styled-table td { padding: 12px 15px; }
    .styled-table tbody tr { border-bottom: 1px solid #dddddd; background-color: #1f2937; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. DATA & AI ENGINE (LOGIC PRESERVED)
# ==========================================

@st.cache_data
def load_and_clean_data():
    files = ["01-15 2.0.csv", "1-15.csv"]
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
    
    if not all_dfs: return None
    return pd.concat(all_dfs).sort_values('Serial').drop_duplicates().reset_index(drop=True)

@st.cache_resource
def train_ai_engine(_data):
    # Feature Engineering (Lags 1-10 + Rolling)
    df = _data.copy()
    for i in range(1, 11):
        df[f'lag_{i}'] = df['Outcome_Num'].shift(i)
    df['rolling_avg_3'] = df['Outcome_Num'].rolling(window=3).mean().shift(1)
    df['rolling_avg_5'] = df['Outcome_Num'].rolling(window=5).mean().shift(1)
    df = df.dropna()
    
    features = [f'lag_{i}' for i in range(1, 11)] + ['rolling_avg_3', 'rolling_avg_5']
    X = df[features]
    y = df['Outcome_Num']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
    
    # Ensemble Models
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    lr = LogisticRegression().fit(X_train, y_train)
    
    acc = accuracy_score(y_test, rf.predict(X_test))
    return (rf, gb, lr), features, acc

# Load Data and Models
data = load_and_clean_data()
if data is not None:
    models, feature_cols, backtest_acc = train_ai_engine(data)
    rf, gb, lr = models

# ==========================================
# 3. SESSION STATE & DASHBOARD
# ==========================================
if 'history' not in st.session_state: st.session_state.history = []
if 'last_pred' not in st.session_state: st.session_state.last_pred = None
if 'streak_n' not in st.session_state: st.session_state.streak_n = 0
if 'streak_t' not in st.session_state: st.session_state.streak_t = ""

st.markdown('<p class="main-title">🤖 SUPER AI PREDICTION ENGINE</p>', unsafe_allow_html=True)

# Dashboard Stats
if st.session_state.history:
    hist_df = pd.DataFrame(st.session_state.history)
    wins = [1 if "WIN" in r['Result'] else 0 for r in st.session_state.history]
    
    # Calculate Max Streaks
    def get_max_streak(binary_list):
        m = c = 0
        for x in binary_list:
            if x == 1: c += 1; m = max(m, c)
            else: c = 0
        return m

    max_w = get_max_streak(wins)
    max_l = get_max_streak([1-x for x in wins])
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔥 MAX WIN STREAK", max_w)
    c2.metric("💀 MAX LOSS STREAK", max_l)
    c3.metric("✅ TOTAL WINS", sum(wins))
    c4.metric("❌ TOTAL LOSSES", len(wins) - sum(wins))

# ==========================================
# 4. PREDICTION LOGIC (FAST & EYE-CATCHING)
# ==========================================
st.subheader("🔢 Manual Prediction Input")
u_num = st.number_input("Last Number (0-9):", 0, 9, 5)

if st.button("🚀 PREDICT NEXT RESULT"):
    curr_t = "BIG" if u_num >= 5 else "SMALL"
    curr_val = 1 if u_num >= 5 else 0
    
    # Streak Logic
    res_str = "START"
    if st.session_state.last_pred:
        otype = "WIN" if st.session_state.last_pred == curr_t else "LOSS"
        if otype == st.session_state.streak_t: st.session_state.streak_n += 1
        else: st.session_state.streak_n = 1; st.session_state.streak_t = otype
        res_str = f"{st.session_state.streak_t} {st.session_state.streak_n}"

    # Engine Logic (Lags)
    lags = list(data['Outcome_Num'].tail(10).values)
    lags.append(curr_val)
    lags = lags[-10:][::-1]
    in_df = pd.DataFrame([lags + [np.mean(lags[:3]), np.mean(lags[:5])]], columns=feature_cols)
    
    # Ensemble Vote
    p = (rf.predict_proba(in_df)[0][1] + gb.predict_proba(in_df)[0][1] + lr.predict_proba(in_df)[0][1]) / 3
    new_pred = "BIG" if p > 0.5 else "SMALL"
    conf = p if p > 0.5 else (1 - p)

    st.session_state.history.insert(0, {
        "Input": u_num, "Type": curr_t, "Result": res_str,
        "Prediction": new_pred, "Confidence": f"{conf*100:.1f}%"
    })
    st.session_state.last_pred = new_pred

# Main Display Card
st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
k1, k2, k3 = st.columns(3)
with k1:
    st.metric("STREAK", f"{st.session_state.streak_t} {st.session_state.streak_n}")
with k2:
    color_class = "big-text" if st.session_state.last_pred == "BIG" else "small-text"
    st.markdown(f"<h3>AI NEXT GUESS</h3><h1 class='{color_class}'>{st.session_state.last_pred if st.session_state.last_pred else '---'}</h1>", unsafe_allow_html=True)
with k3:
    st.metric("BACKTEST ACCURACY", f"{backtest_acc*100:.2f}%")
st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 5. BATCH EVALUATION & HISTORY
# ==========================================
tab1, tab2 = st.tabs(["📜 Manual History", "📂 Batch Evaluation"])

with tab1:
    if st.session_state.history:
        # Custom HTML Table for Professional Colors
        html = '<table class="styled-table"><thead><tr><th>Input</th><th>Type</th><th>Result</th><th>Next Pred</th><th>Conf</th></tr></thead><tbody>'
        for h in st.session_state.history:
            t_color = "big-text" if h['Type'] == "BIG" else "small-text"
            p_color = "big-text" if h['Prediction'] == "BIG" else "small-text"
            res_color = "white"
            if "WIN" in h['Result']: res_color = "#39FF14"
            elif "LOSS" in h['Result']: res_color = "#FF3131"
            
            html += f"<tr><td>{h['Input']}</td><td class='{t_color}'>{h['Type']}</td><td style='color:{res_color}'>{h['Result']}</td><td class='{p_color}'>{h['Prediction']}</td><td>{h['Confidence']}</td></tr>"
        html += '</tbody></table>'
        st.markdown(html, unsafe_allow_html=True)
        if st.button("Reset Session"):
            st.session_state.history = []; st.session_state.last_pred = None; st.rerun()

with tab2:
    eval_file = st.file_uploader("Upload Testing CSV (Must have '0 to 9' column)", type="csv")
    if eval_file:
        edf = pd.read_csv(eval_file)
        # Logic for Batch Eval (Similar to manual but loop)
        results = []
        el_pred = None
        es_n = 0; es_t = ""
        e_hist = list(data['Outcome_Num'].tail(10).values)
        
        for i, row in edf.iterrows():
            n = row['0 to 9']
            act_t = "BIG" if n >= 5 else "SMALL"
            act_v = 1 if n >= 5 else 0
            
            eres = "N/A"
            if el_pred:
                et = "WIN" if el_pred == act_t else "LOSS"
                if et == es_t: es_n += 1
                else: es_n = 1; es_t = et
                eres = f"{es_t} {es_n}"
            
            e_hist.append(act_v)
            cur_l = e_hist[-10:][::-1]
            in_df = pd.DataFrame([cur_l + [np.mean(cur_l[:3]), np.mean(cur_l[:5])]], columns=feature_cols)
            ep = (rf.predict_proba(in_df)[0][1] + gb.predict_proba(in_df)[0][1]) / 2
            el_pred = "BIG" if ep > 0.5 else "SMALL"
            
            results.append({"Game": i+1, "Number": n, "Actual": act_t, "Result": eres, "AI_Next": el_pred})
        
        res_df = pd.DataFrame(results)
        st.dataframe(res_df, use_container_width=True)
        st.download_button("📥 Download Eval CSV", res_df.to_csv(index=False), "eval.csv", "text/csv")
