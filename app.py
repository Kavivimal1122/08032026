import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

def run_prediction_engine(file_paths):
    print("--- 1. CSV Data Loading & 2. Cleaning ---")
    all_dfs = []
    for file in file_paths:
        df = pd.read_csv(file)
        # Normalize columns (some files use B/S, others S/B)
        df.columns = [c.strip() for c in df.columns]
        target_col = [c for c in df.columns if 'B/S' in c or 'S/B' in c][0]
        df.rename(columns={target_col: 'Outcome'}, inplace=True)
        all_dfs.append(df[['Ser No', 'Outcome']])
    
    # Merge and Sort by Serial Number
    full_df = pd.concat(all_dfs).sort_values(by='Ser No').reset_index(drop=True)
    # Map Outcome: Big (B) = 1, Small (S) = 0
    full_df['Outcome_Num'] = full_df['Outcome'].map({'B': 1, 'S': 0})
    full_df = full_df.dropna()
    print(f"Total Records Cleaned: {len(full_df)}")

    print("\n--- 3. Feature Engineering ---")
    # We look at the last 5 results (Lags) and a Rolling Average to find patterns
    lags = 5
    for i in range(1, lags + 1):
        full_df[f'lag_{i}'] = full_df['Outcome_Num'].shift(i)
    
    # Calculate rolling average (momentum of previous 3)
    full_df['rolling_avg_3'] = full_df['Outcome_Num'].rolling(window=3).mean().shift(1)
    full_df = full_df.dropna()

    # Define Features (X) and Target (y)
    features = [f'lag_{i}' for i in range(1, lags + 1)] + ['rolling_avg_3']
    X = full_df[features]
    y = full_df['Outcome_Num']

    print("\n--- 4. Train Models & 5. Evaluate (Self-Test) ---")
    # Split 80% for training, 20% for testing (Self-Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Model 1: Random Forest (Pattern Recognition)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))

    # Model 2: Gradient Boosting (Trend Analysis)
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train)
    gb_acc = accuracy_score(y_test, gb.predict(X_test))

    # Model 3: Logistic Regression (Probability Baseline)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test))

    print(f"Self-Test Accuracy (RF): {rf_acc*100:.2f}%")
    print(f"Self-Test Accuracy (GB): {gb_acc*100:.2f}%")

    print("\n--- 7. Ensemble Model (Weighted Prediction) ---")
    # Calculate the features for the very NEXT prediction
    last_row = full_df.iloc[-1]
    next_features = {
        'lag_1': last_row['Outcome_Num'],
        'lag_2': last_row['lag_1'],
        'lag_3': last_row['lag_2'],
        'lag_4': last_row['lag_3'],
        'lag_5': last_row['lag_4'],
        'rolling_avg_3': (last_row['Outcome_Num'] + last_row['lag_1'] + last_row['lag_2']) / 3
    }
    X_next = pd.DataFrame([next_features])[features]

    # Average the probabilities from all models
    prob_rf = rf.predict_proba(X_next)[0][1]
    prob_gb = gb.predict_proba(X_next)[0][1]
    prob_lr = lr.predict_proba(X_next)[0][1]
    
    ensemble_prob = (prob_rf + prob_gb + prob_lr) / 3

    print("\n--- 8. Prediction Engine & 9. Confidence Score ---")
    prediction = "BIG (B)" if ensemble_prob > 0.5 else "SMALL (S)"
    confidence = ensemble_prob if ensemble_prob > 0.5 else (1 - ensemble_prob)

    print("=" * 40)
    print(f"FINAL PREDICTION: {prediction}")
    print(f"CONFIDENCE SCORE: {confidence * 100:.2f}%")
    print("=" * 40)

# Run the system with your two files
file_list = ['01-15 2.0.csv', '1-15.csv']
run_prediction_engine(file_list)
