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
# 1. PROFESSIONAL UI & THEME (NEON STYLE)
# ==========================================
st.set_page_config(page_title="Super AI Prediction Pro", layout="wide", page_icon="🔮")

st.markdown("""
    <style>
    /* Main Dark Theme */
    .stApp { background-color: #0e1117; color: #ffffff; }
    
    /* Neon Titles */
    .main-title { 
        font-size: 50px; font-weight: 900; text-align: center; 
        background: -webkit-linear-gradient(#00d4ff, #0055ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    
    /* High-Contrast Colors */
    .big-text { color: #FF3131 !important; font-weight: bold; text-shadow: 0 0 10px rgba(255, 49, 49, 0.5); }
    .small-text { color: #39FF14 !important; font-weight: bold; text-shadow: 0 0 10px rgba(57, 255, 20, 0.5); }
    
    /* Result Display Cards */
    .prediction-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 30px; border-radius: 20px; text-align: center;
        border: 2px solid rgba(255, 255, 255, 0.1);
        margin-top: 20px;
    }
    
    /* Tables */
    .styled-table { width: 100%; color: white; border-collapse: collapse; margin-top: 20px; }
    .styled-table th { background-color: #1f2937; padding: 12px; border: 1px solid #374151; }
    .styled-table td { padding: 12px; border: 1px solid #374151; text-align: center; }

    /* Fast Button */
    .stButton>button {
        width: 100%; border-radius: 10px; height: 50px; 
        background-color: #00d4ff; color: black; font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 10-STEP AI PIPELINE (PRESERVED LOGIC)
# ==========================================

@st.cache_resource
def initialize_ai_engine():
    # 1. CSV Data & 2. Clean Data
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
    
    full_data = pd.concat(all_dfs).sort_values('Serial').drop_duplicates().reset_index(drop=True)

    # 3. Feature Engineering (10 Lags + Momentum)
    df_feat = full_data.copy()
    for i in range(1, 11):
        df_feat[f'lag_{i}'] = df_feat['Outcome_Num'].shift(i)
    df_feat['rolling_avg_3'] = df_feat['Outcome_Num'].rolling(window=3).mean().shift(1)
    df_feat['rolling_avg_5'] = df_feat['Outcome_Num'].rolling(window=5).mean().shift(1)
    df_feat = df_feat.dropna()
    
    features = [f'lag_{i}' for i in range(1, 11)] + ['rolling_avg_3', 'rolling_avg_5']
    X = df_feat[features]
    y = df_feat['Outcome_Num']

    # 4. Train Models & 5, 6, 7. Evaluation/Self Test/Backtest
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
    rf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42).fit(X_train, y_train)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    
    acc = accuracy_score(y_test, rf.predict(X_test))
    return full_data, (rf, gb), features, acc

# Load AI
full_data, models, feature_cols, backtest_acc = initialize_ai_engine()

# ==========================================
# 3. SESSION STATE (FIXES CONTEXT ERRORS)
# ==========================================
if 'history' not in st.session_state: st.session_state.history = []
if 'last_pred' not in st.session_state: st.session_state.last_pred = None
if 'streak_n' not in st.session_state: st.session_state.streak_n = 0
if 'streak_t' not in st.session_state: st.session_state.streak_t = ""
# Maintains session memory to match batch evaluation logic
if 'memory' not in st.session_state: 
    st.session_state.memory = list(full_data['Outcome_Num'].tail(10).values)

st.markdown('<p class="main-title">🔮 SUPER AI PREDICTION ENGINE</p>', unsafe_allow_html=True)

# ==========================================
# 4. DASHBOARD
# ==========================================
if st.session_state.history:
    res_vals = [1 if "WIN" in h['Result'] else 0 for h in st.session_state.history]
    total_w = sum(res_vals)
    total_l = len(res_vals) - total_w
    
    # Calculate Max Streaks from history
    def get_max_streak(bits):
        m = c = 0
        for b in bits:
            if b == 1: c += 1; m = max(m, c)
            else: c = 0
        return m

    mw = get_max_streak(res_vals[::-1])
    ml = get_max_streak([1-x for x in res_vals][::-1])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔥 MAX WIN STREAK", mw)
    c2.metric("💀 MAX LOSS STREAK", ml)
    c3.metric("✅ TOTAL WINS", total_w)
    c4.metric("❌ TOTAL LOSSES", total_l)

# ==========================================
# 5. PREDICTION & MANUAL INPUT
# ==========================================
st.divider()
col_in, col_disp = st.columns([1, 2])

with col_in:
    st.subheader("🔢 Manual Input")
    user_num = st.number_input("Last Game Number (0-9):", 0, 9, 5)
    
    if st.button("🚀 SUBMIT & ANALYZE"):
        actual_t = "BIG" if user_num >= 5 else "SMALL"
        actual_v = 1 if user_num >= 5 else 0
        
        # 1. Update Result/Streak
        res_str = "START"
        if st.session_state.last_pred:
            res_t = "WIN" if st.session_state.last_pred == actual_t else "LOSS"
            if res_t == st.session_state.streak_t: st.session_state.streak_n += 1
            else: st.session_state.streak_n = 1; st.session_state.streak_t = res_t
            res_str = f"{st.session_state.streak_t} {st.session_state.streak_n}"

        # 2. Add to AI Memory
        st.session_state.memory.append(actual_v)
        
        # 3. Ensemble Prediction (8. Ensemble & 9. Prediction Engine)
        lags = st.session_state.memory[-10:][::-1]
        in_df = pd.DataFrame([lags + [np.mean(lags[:3]), np.mean(lags[:5])]], columns=feature_cols)
        
        prob = (models[0].predict_proba(in_df)[0][1] + models[1].predict_proba(in_df)[0][1]) / 2
        next_p = "BIG" if prob > 0.5 else "SMALL"
        conf = prob if prob > 0.5 else (1 - prob)

        # 4. Save History (10. Confidence Score)
        st.session_state.history.insert(0, {
            "Input": user_num, "Type": actual_t, "Result": res_str,
            "Next Pred": next_p, "Confidence": f"{conf*100:.2f}%"
        })
        st.session_state.last_pred = next_p

with col_disp:
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    if st.session_state.last_pred:
        color = "#FF3131" if st.session_state.last_pred == "BIG" else "#39FF14"
        st.markdown(f"<h3>AI NEXT GUESS</h3><h1 style='color:{color}; font-size: 80px;'>{st.session_state.last_pred}</h1>", unsafe_allow_html=True)
    else:
        st.markdown("<h3>Ready for first input...</h3><h1 style='color:gray;'>---</h1>", unsafe_allow_html=True)
    st.markdown(f"**Backtest Win Rate:** {backtest_acc*100:.2f}%", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 6. HISTORY & BATCH EVALUATION
# ==========================================
tab1, tab2 = st.tabs(["📜 Manual Win/Loss Log", "📂 Batch Evaluation"])

with tab1:
    if st.session_state.history:
        html = '<table class="styled-table"><tr><th>Input</th><th>Type</th><th>Result</th><th>Next Prediction</th><th>Conf</th></tr>'
        for h in st.session_state.history:
            t_cls = "big-text" if h['Type'] == "BIG" else "small-text"
            p_cls = "big-text" if h['Next Pred'] == "BIG" else "small-text"
            r_style = "color:#39FF14;" if "WIN" in h['Result'] else "color:#FF3131;"
            html += f"<tr><td>{h['Input']}</td><td class='{t_cls}'>{h['Type']}</td><td style='{r_style}'>{h['Result']}</td><td class='{p_cls}'>{h['Next Pred']}</td><td>{h['Confidence']}</td></tr>"
        html += "</table>"
        st.markdown(html, unsafe_allow_html=True)
        if st.button("Reset Session"):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()

with tab2:
    eval_f = st.file_uploader("Upload Evaluation CSV ('0 to 9' column)", type="csv")
    if eval_f:
        edf = pd.read_csv(eval_f)
        results = []
        e_pred = None
        es_n = 0; es_t = ""
        e_mem = list(full_data['Outcome_Num'].tail(10).values)
        
        for i, row in edf.iterrows():
            num = row['0 to 9']
            at = "BIG" if num >= 5 else "SMALL"; av = 1 if num >= 5 else 0
            
            # Result check
            estr = "N/A"
            if e_pred:
                et = "WIN" if e_pred == at else "LOSS"
                if et == es_t: es_n += 1
                else: es_n = 1; es_t = et
                estr = f"{es_t} {es_n}"
            
            # Predict Next
            e_mem.append(av)
            lags = e_mem[-10:][::-1]
            in_df = pd.DataFrame([lags + [np.mean(lags[:3]), np.mean(lags[:5])]], columns=feature_cols)
            ep = (models[0].predict_proba(in_df)[0][1] + models[1].predict_proba(in_df)[0][1]) / 2
            e_pred = "BIG" if ep > 0.5 else "SMALL"
            
            results.append({"Game": i+1, "Number": num, "Actual": at, "Result": estr, "AI_Next": e_pred})
        
        res_df = pd.DataFrame(results)
        st.dataframe(res_df, use_container_width=True)
        st.download_button("📥 Download Full Evaluation Report", res_df.to_csv(index=False), "ai_report.csv", "text/csv")
