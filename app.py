import pandas as pd
from collections import Counter

class AIStudent:
    def __init__(self):
        self.data_history = []
        self.transitions = {}
        self.win_count = 0
        self.total_attempts = 0
        self.streak = 0
        self.max_streak = 0

    def _build_model(self):
        self.transitions = {}
        for i in range(len(self.data_history) - 1):
            current = self.data_history[i]
            next_val = self.data_history[i+1]
            if current not in self.transitions:
                self.transitions[current] = []
            self.transitions[current].append(next_val)

    def predict_next(self, current_number):
        if current_number not in self.transitions:
            return None, 0, []
        
        possibilities = self.transitions[current_number]
        counts = Counter(possibilities)
        total = len(possibilities)
        
        # Get top 3 predictions for the style
        top_3 = counts.most_common(3)
        prediction = top_3[0][0]
        confidence = (top_3[0][1] / total) * 100
        
        return prediction, confidence, top_3

    def dashboard(self, current_number, actual_next=None):
        prediction, confidence, top_3 = self.predict_next(current_number)
        
        # 1. STYLE: The Header
        print("\n" + "╔" + "═"*45 + "╗")
        print(f"║ {'🧠 AI STUDENT INTELLIGENCE TERMINAL':^43} ║")
        print("╠" + "═"*45 + "╣")

        # 2. LOGIC: Probability Breakdown
        print(f"║  INPUT NUMBER: {current_number:<28} ║")
        if prediction:
            bar_length = int(confidence / 5)
            meter = "█" * bar_length + "░" * (20 - bar_length)
            print(f"║  PREDICTION:   {prediction:<28} ║")
            print(f"║  CONFIDENCE:   [{meter}] {confidence:>3.0f}% ║")
        else:
            print(f"║  STATUS:       Learning new pattern...      ║")

        # 3. EVALUATION: Performance Tracking
        if actual_next is not None:
            self.total_attempts += 1
            is_correct = (prediction == actual_next)
            
            if is_correct:
                self.win_count += 1
                self.streak += 1
                self.max_streak = max(self.max_streak, self.streak)
                result_text = "✅ WIN STREAK: " + str(self.streak)
            else:
                self.streak = 0
                result_text = f"❌ MISS (Actual: {actual_next})"

            win_rate = (self.win_count / self.total_attempts) * 100
            
            print("╟" + "─"*45 + "╢")
            print(f"║  LAST RESULT:  {result_text:<28} ║")
            print(f"║  ACCURACY:     {win_rate:>6.2f}% | BEST STREAK: {self.max_streak:<3} ║")

        print("╚" + "═"*45 + "╝\n")

# Simulation
ai = AIStudent()
ai.data_history = [1, 2, 3, 1, 2, 4, 1, 2, 3, 5, 1, 2, 3] # Sample Data
ai._build_model()

# Test run
ai.dashboard(3, actual_next=1) # The AI sees '3' and predicts what comes after it in history
