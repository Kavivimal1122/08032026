import pandas as pd
from collections import defaultdict

def predict_next_outcome(file_paths):
    all_data = []
    
    # 1. Load and combine the CSV files
    for file in file_paths:
        try:
            df = pd.read_csv(file)
            # Find the column that represents Big/Small (B/S)
            # Looking at your snippets, the column names vary slightly
            col_name = [c for c in df.columns if 'B/S' in c or 'S/B' in c][0]
            all_data.extend(df[col_name].dropna().tolist())
        except Exception as e:
            print(f"Error loading {file}: {e}")

    if not all_data:
        return "No data found."

    # 2. Build the Transition Matrix (Markov Chain)
    transitions = defaultdict(list)
    for i in range(len(all_data) - 1):
        current_state = all_data[i]
        next_state = all_data[i+1]
        transitions[current_state].append(next_state)

    # 3. Get the very last result from your data to predict the NEXT one
    last_result = all_data[-1]
    
    # 4. Calculate Probabilities
    next_options = transitions[last_result]
    if not next_options:
        return f"Last result was {last_result}, but no historical data follows it."

    big_count = next_options.count('B')
    small_count = next_options.count('S')
    total = len(next_options)

    prob_big = (big_count / total) * 100
    prob_small = (small_count / total) * 100

    # 5. Output Result
    prediction = "BIG (B)" if prob_big > prob_small else "SMALL (S)"
    if prob_big == prob_small: prediction = "TIE (Neutral)"

    print("-" * 30)
    print(f"LATEST RESULT IN DATA: {last_result}")
    print("-" * 30)
    print(f"Probability of BIG: {prob_big:.2f}%")
    print(f"Probability of SMALL: {prob_small:.2f}%")
    print("-" * 30)
    print(f">>> PREDICTED NEXT: {prediction} <<<")
    print("-" * 30)

# Run the function with your files
files = ['1-15.csv', '01-15 2.0.csv']
predict_next_outcome(files)
