import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI Prediction Engine", page_icon="📈")

st.title("🛡️ Big/Small Prediction Pipeline")

# Helper to find any CSV files uploaded to GitHub
def get_csv_files():
    return [f for f in os.listdir('.') if f.endswith('.csv')]

def run_pipeline():
    csv_files = get_csv_files()
    
    if not csv_files:
        st.error("Step 1 Failed: No CSV files found in your GitHub repository.")
        return

    # --- 1. CSV Data & 2. Clean Data ---
    all_dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        # Identify columns dynamically
        target_col = [c for c in df.columns if 'B/S' in c or 'S/B' in c][0]
        ser_col = [c for c in df.columns if 'Ser' in c or 'Serial' in c][0]
        
        df = df.rename(columns={target_col: 'Outcome', ser_col: 'Serial'})
        df['Outcome_Num'] = df['Outcome'].map({'B': 1, 'S': 0})
        all_dfs.append(df[['Serial', 'Outcome_Num']])
    
    full_data = pd.concat(all_dfs).sort_values('Serial').drop_duplicates().dropna()
    
    # --- 3. Feature Engineering ---
    for i in range(1, 6):
        full_data[f'lag_{i}'] = full_data['Outcome_Num'].shift(i)
    full_data['rolling_avg_3'] = full_data['Outcome_Num'].rolling(window=3).mean().shift(1)
    full_data = full_data.dropna()
    
    features = [f'lag_{i}' for i in range(1, 6)] + ['rolling_avg_3']
    X = full_data[features]
    y = full_data['Outcome_Num']

    # --- 4. Train Models & 5. Evaluate ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    gb = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)
    lr = LogisticRegression().fit(X_train, y_train)
    
    # --- 6. Self Test & 7. Backtest ---
    acc = accuracy_score(y_test, rf.predict(X_test))
    
    # --- 8. Ensemble Model & 9. Prediction Engine ---
    last_row = full_data.iloc[-1]
    next_x = pd.DataFrame([{
        'lag_1': last_row['Outcome_Num'],
        'lag_2': last_row['lag_1'],
        'lag_3': last_row['lag_2'],
        'lag_4': last_row['lag_3'],
        'lag_5': last_row['lag_4'],
        'rolling_avg_3': (last_row['Outcome_Num'] + last_row['lag_1'] + last_row['lag_2']) / 3
    }])[features]

    # Weighted Ensemble
    prob = (rf.predict_proba(next_x)[0][1] + gb.predict_proba(next_x)[0][1] + lr.predict_proba(next_x)[0][1]) / 3
    
    # --- 10. Confidence Score ---
    prediction = "BIG (B)" if prob > 0.5 else "SMALL (S)"
    confidence = prob if prob > 0.5 else (1 - prob)

    # UI Output
    st.success("Pipeline Analysis Complete")
    
    col1, col2 = st.columns(2)
    col1.metric("NEXT TARGET", prediction)
    col2.metric("CONFIDENCE", f"{confidence*100:.2f}%")
    
    st.info(f"Analysis based on {len(full_data)} historical records. Backtest Accuracy: {acc*100:.2f}%")

if st.button("EXECUTE 10-STEP PREDICTION FLOW"):
    run_pipeline()
else:
    st.write("Click the button to run the full AI ensemble.")
