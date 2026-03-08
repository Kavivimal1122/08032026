import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- Page Layout ---
st.set_page_config(page_title="Big/Small AI Engine", layout="centered")
st.title("🎰 Automated Prediction Pipeline")

# File Configuration
files = ["01-15 2.0.csv", "1-15.csv"]

def run_prediction_flow():
    # 1. & 2. CSV Data & Clean Data
    with st.status("🚀 Running Prediction Pipeline...", expanded=True) as status:
        st.write("Reading CSV files from GitHub...")
        dfs = []
        for f in files:
            if not os.path.exists(f):
                st.error(f"File {f} not found in repository!")
                return
            df = pd.read_csv(f)
            df.columns = [c.strip() for c in df.columns]
            # Standardize Outcome Column
            col = [c for c in df.columns if 'B/S' in c or 'S/B' in c][0]
            df = df.rename(columns={col: 'Target', 'Ser No': 'Serial'})
            df['Target_Num'] = df['Target'].map({'B': 1, 'S': 0})
            dfs.append(df[['Serial', 'Target_Num']])
        
        full_data = pd.concat(dfs).sort_values('Serial').drop_duplicates().reset_index(drop=True)
        st.write(f"✅ Cleaned {len(full_data)} records.")

        # 3. Feature Engineering
        st.write("🛠️ Engineering Features...")
        for i in range(1, 6):
            full_data[f'lag_{i}'] = full_data['Target_Num'].shift(i)
        full_data['mva'] = full_data['Target_Num'].rolling(window=3).mean().shift(1)
        full_data = full_data.dropna()
        
        features = [f'lag_{i}' for i in range(1, 6)] + ['mva']
        X = full_data[features]
        y = full_data['Target_Num']

        # 4, 5, 6, 7. Train, Evaluate, Self Test, Backtest
        st.write("🧠 Training Ensemble Models...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
        gb = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)
        
        test_acc = accuracy_score(y_test, rf.predict(X_test))
        st.write(f"📈 Backtest Accuracy: {test_acc*100:.2f}%")

        # 8, 9, 10. Ensemble, Engine, Confidence
        st.write("🔮 Generating Prediction...")
        last = full_data.iloc[-1]
        input_data = pd.DataFrame([{
            'lag_1': last['Target_Num'], 'lag_2': last['lag_1'], 'lag_3': last['lag_2'],
            'lag_4': last['lag_3'], 'lag_5': last['lag_4'],
            'mva': (last['Target_Num'] + last['lag_1'] + last['lag_2']) / 3
        }])[features]

        prob = (rf.predict_proba(input_data)[0][1] + gb.predict_proba(input_data)[0][1]) / 2
        prediction = "BIG (B)" if prob > 0.5 else "SMALL (S)"
        conf = prob if prob > 0.5 else (1 - prob)
        
        status.update(label="✅ Prediction Complete!", state="complete", expanded=False)

    # Big Result Display
    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("TARGET PREDICTION", prediction)
    c2.metric("CONFIDENCE SCORE", f"{conf*100:.2f}%")
    st.divider()
    
    st.success(f"Pipeline finished successfully. Based on historical patterns, the next target is likely {prediction}.")

# Auto-run the function
run_prediction_flow()
