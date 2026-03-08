import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time

st.set_page_config(page_title="Big/Small Prediction Engine", layout="wide")
st.title("🚀 AI Prediction Pipeline")

def run_pipeline():
    # --- 1. CSV Data ---
    st.subheader("📥 1. CSV Data & Cleaning")
    try:
        file1 = "01-15 2.0.csv"
        file2 = "1-15.csv"
        
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        
        # Standardizing columns
        def clean_df(df):
            df.columns = [c.strip() for c in df.columns]
            target = [c for c in df.columns if 'B/S' in c or 'S/B' in c][0]
            df = df.rename(columns={target: 'Outcome', 'Ser No': 'Serial'})
            df['Outcome_Num'] = df['Outcome'].map({'B': 1, 'S': 0})
            return df[['Serial', 'Outcome_Num']]

        data = pd.concat([clean_df(df1), clean_df(df2)]).sort_values('Serial').drop_duplicates().dropna()
        st.write(f"✅ Data Loaded: {len(data)} records found.")
        
        # --- 2. Feature Engineering ---
        st.subheader("⚙️ 2. Feature Engineering")
        lags = 5
        for i in range(1, lags + 1):
            data[f'lag_{i}'] = data['Outcome_Num'].shift(i)
        data['rolling_avg'] = data['Outcome_Num'].rolling(window=3).mean().shift(1)
        data = data.dropna()
        
        features = [f'lag_{i}' for i in range(1, lags + 1)] + ['rolling_avg']
        X = data[features]
        y = data['Outcome_Num']
        st.write("✅ Features generated (Lags 1-5 + Rolling Average).")

        # --- 3. Train Models & 4. Evaluate (Self Test) ---
        st.subheader("🧠 3. Model Training & 4. Self Test")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
        gb = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)
        lr = LogisticRegression().fit(X_train, y_train)
        
        rf_acc = accuracy_score(y_test, rf.predict(X_test))
        gb_acc = accuracy_score(y_test, gb.predict(X_test))
        st.write(f"✅ RF Accuracy: {rf_acc*100:.2f}% | GB Accuracy: {gb_acc*100:.2f}%")

        # --- 5. Backtest ---
        st.subheader("📊 5. Backtest Results")
        test_preds = (rf.predict_proba(X_test)[:, 1] + gb.predict_proba(X_test)[:, 1]) / 2
        backtest_acc = accuracy_score(y_test, (test_preds > 0.5).astype(int))
        st.info(f"Historical Backtest Win Rate: {backtest_acc*100:.2f}%")

        # --- 6. Ensemble Model & Prediction Engine ---
        st.subheader("🔮 6. Prediction Engine")
        last_row = data.iloc[-1]
        next_features = pd.DataFrame([{
            'lag_1': last_row['Outcome_Num'],
            'lag_2': last_row['lag_1'],
            'lag_3': last_row['lag_2'],
            'lag_4': last_row['lag_3'],
            'lag_5': last_row['lag_4'],
            'rolling_avg': (last_row['Outcome_Num'] + last_row['lag_1'] + last_row['lag_2']) / 3
        }])[features]

        # Ensemble Logic
        prob_rf = rf.predict_proba(next_features)[0][1]
        prob_gb = gb.predict_proba(next_features)[0][1]
        prob_lr = lr.predict_proba(next_features)[0][1]
        
        final_prob = (prob_rf + prob_gb + prob_lr) / 3
        
        # --- 7. Confidence Score ---
        prediction = "BIG (B)" if final_prob > 0.5 else "SMALL (S)"
        confidence = final_prob if final_prob > 0.5 else (1 - final_prob)

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.metric("NEXT PREDICTION", prediction)
        with c2:
            st.metric("CONFIDENCE SCORE", f"{confidence*100:.2f}%")
            
    except Exception as e:
        st.error(f"⚠️ Error: Ensure '01-15 2.0.csv' and '1-15.csv' are in your GitHub folder. Error: {e}")

if st.button("RUN AUTO-PREDICTION ENGINE"):
    with st.spinner("Processing Pipeline..."):
        run_pipeline()
else:
    st.info("Click the button above to start the analysis.")
