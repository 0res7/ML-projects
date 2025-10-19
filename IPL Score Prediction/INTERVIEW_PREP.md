# Interview Preparation: IPL Score Prediction

## 1. Project Overview

**Problem Statement:** Predict the final first innings score in an Indian Premier League (IPL) cricket match based on current match situation (overs, runs, wickets, venue, teams).

**Objective:** Build regression models that provide real-time score predictions during live matches, useful for broadcasters, betting platforms, and fan engagement.

**Use Cases:**
- Live match commentary and graphics
- Sports betting odds calculation
- Team strategy optimization
- Fan engagement and predictions

---

## 2. Technical Concepts

### Regression Problem
- **Target:** Total runs at end of first innings (80-250 range)
- **Dynamic Prediction:** Predictions update as match progresses
- **Temporal Features:** Current overs, run rate, wickets

### Algorithms
- **Linear Regression:** Baseline model
- **Ridge/Lasso Regression:** Regularized linear models
- **Random Forest Regressor:** Ensemble method
- **XGBoost/LightGBM:** Gradient boosting
- **Neural Networks:** Deep learning approach

---

## 3. Mathematical Foundations

### Linear Regression
\[
\text{Final Score} = \beta_0 + \beta_1 \times \text{current\_runs} + \beta_2 \times \text{overs} + ... + \epsilon
\]

### Run Rate Calculation
\[
\text{Current Run Rate} = \frac{\text{Runs Scored}}{\text{Overs Completed}}
\]

### Projected Score (Simple)
\[
\text{Projected Score} = \text{Current Runs} + \text{Run Rate} \times \text{Overs Remaining}
\]

### Evaluation Metrics

**Mean Absolute Error:**
\[
MAE = \frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|
\]

**Root Mean Squared Error:**
\[
RMSE = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}
\]

**R² Score:**
\[
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
\]

---

## 4. Implementation Details

### Feature Engineering

**Key Features:**
```python
# Match Context
- batting_team (encoded)
- bowling_team (encoded)  
- venue (encoded)

# Current Match State  
- overs (0-20)
- runs (cumulative)
- wickets (0-10)
- runs_last_5 (recent performance)
- wickets_last_5

# Derived Features
- current_run_rate = runs / overs
- balls_remaining = (20 - overs) * 6
- wickets_remaining = 10 - wickets
```

### Workflow
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('ipl.csv')

# Data cleaning
# Remove mid, venue, batsman, bowler columns (too specific)
df = df.drop(['mid', 'batsman', 'bowler', 'striker', 'non-striker'], axis=1)

# Keep only consistent teams (appeared in multiple seasons)
consistent_teams = ['Mumbai Indians', 'Kolkata Knight Riders', 
                   'Royal Challengers Bangalore', 'Chennai Super Kings',
                   'Rajasthan Royals', 'Delhi Capitals', 
                   'Kings XI Punjab', 'Sunrisers Hyderabad']
df = df[df['batting_team'].isin(consistent_teams)]
df = df[df['bowling_team'].isin(consistent_teams)]

# Remove first 5 overs (powerplay, different dynamics)
df = df[df['overs'] >= 5.0]

# Convert date
df['date'] = pd.to_datetime(df['date'])

# Feature engineering
df['current_run_rate'] = df['runs'] / df['overs']
df['balls_remaining'] = (20 - df['overs']) * 6
df['wickets_remaining'] = 10 - df['wickets']
df['crr_multiply_remaining'] = df['current_run_rate'] * df['balls_remaining'] / 6

# Encode categorical variables
le_batting = LabelEncoder()
le_bowling = LabelEncoder()
le_venue = LabelEncoder()

df['batting_team_encoded'] = le_batting.fit_transform(df['batting_team'])
df['bowling_team_encoded'] = le_bowling.fit_transform(df['bowling_team'])
df['venue_encoded'] = le_venue.fit_transform(df['venue'])

# Prepare data
features = ['batting_team_encoded', 'bowling_team_encoded', 'venue_encoded',
           'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5',
           'current_run_rate', 'balls_remaining', 'wickets_remaining']

X = df[features]
y = df['total']  # Final score

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{name}:")
    print(f"MAE: {mae:.2f} runs")
    print(f"RMSE: {rmse:.2f} runs")
    print(f"R²: {r2:.4f}")
    
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R²': r2}

# Feature importance
importances = pd.DataFrame({
    'feature': features,
    'importance': models['Random Forest'].feature_importances_
}).sort_values('importance', ascending=False)
print(importances)

# Visualize predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.title('IPL Score Prediction')
plt.show()
```

---

## 5. Outcomes & Results

### Typical Performance
- **MAE:** 10-15 runs
- **RMSE:** 15-20 runs
- **R²:** 0.90-0.95

### Most Important Features
1. **Current Runs:** Strong positive correlation with final score
2. **Current Run Rate:** Indicates scoring momentum
3. **Wickets Remaining:** More wickets → higher score potential
4. **Runs Last 5 Overs:** Recent form important
5. **Batting Team:** Team strength varies

### Insights
- **Death Overs (15-20):** Highest scoring potential
- **Powerplay Exclusion:** First 5 overs have different dynamics
- **Venue Impact:** Some grounds batting-friendly
- **Team Matchups:** Certain teams perform better against others

---

## 6. Interview Questions & Answers

**Q1: Why remove first 5 overs from training data?**

**A1:** **Powerplay Has Different Dynamics**
- **Powerplay (Overs 1-6):** Only 2 fielders outside circle, aggressive batting
- **Middle Overs (7-15):** Strategic batting, singles and doubles
- **Death Overs (16-20):** Maximum aggression

Including powerplay data adds noise to middle/death over predictions.

**Alternative:** Separate models for different phases

**Q2: How do you handle team name encoding?**

**A2:**
```python
# Label Encoding
le = LabelEncoder()
df['batting_team'] = le.fit_transform(df['batting_team'])
# Mumbai Indians → 0, CSK → 1, etc.

# One-Hot Encoding (better for tree models)
df = pd.get_dummies(df, columns=['batting_team', 'bowling_team'], drop_first=True)

# Target Encoding (advanced)
team_avg_scores = df.groupby('batting_team')['total'].mean()
df['batting_team_avg'] = df['batting_team'].map(team_avg_scores)
```

**Q3: Why is current run rate the most important feature?**

**A3:** **Strong Predictive Power**

**Logic:**
- Team scoring at 10 runs/over likely to continue high scoring
- Team at 6 runs/over unlikely to suddenly accelerate to 12

**Mathematical:**
```
Final Score ≈ Current Runs + (Run Rate × Remaining Overs)
```

**Limitation:** Doesn't account for wickets, acceleration in death overs

**Better Feature:**
```python
# Weighted run rate (recent overs weighted more)
df['weighted_rr'] = (
    0.5 * df['runs_last_5'] / 5 +  # Recent
    0.3 * df['runs_last_10'] / 10 +  # Medium
    0.2 * df['current_run_rate']  # Overall
)
```

**Q4: How would you make predictions for a live match?**

**A4:**

```python
def predict_live_score(batting_team, bowling_team, venue, 
                      overs, runs, wickets, runs_last_5, wickets_last_5):
    """
    Predict final score for ongoing match.
    
    Args:
        batting_team: str
        bowling_team: str
        venue: str
        overs: float (e.g., 12.3 = 12 overs 3 balls)
        runs: int (current score)
        wickets: int (wickets fallen)
        runs_last_5: int (runs in last 5 overs)
        wickets_last_5: int (wickets in last 5 overs)
    
    Returns:
        predicted_score: int
        confidence_interval: (low, high)
    """
    # Encode teams and venue
    batting_encoded = le_batting.transform([batting_team])[0]
    bowling_encoded = le_bowling.transform([bowling_team])[0]
    venue_encoded = le_venue.transform([venue])[0]
    
    # Compute features
    current_run_rate = runs / overs
    balls_remaining = (20 - overs) * 6
    wickets_remaining = 10 - wickets
    crr_multiply = current_run_rate * balls_remaining / 6
    
    # Create feature vector
    features = pd.DataFrame([[
        batting_encoded, bowling_encoded, venue_encoded,
        overs, runs, wickets, runs_last_5, wickets_last_5,
        current_run_rate, balls_remaining, wickets_remaining
    ]], columns=feature_names)
    
    # Predict
    predicted_score = model.predict(features)[0]
    
    # Confidence interval (from Random Forest)
    predictions = [tree.predict(features)[0] for tree in rf.estimators_]
    lower = np.percentile(predictions, 5)
    upper = np.percentile(predictions, 95)
    
    return int(predicted_score), (int(lower), int(upper))

# Example usage during live match
# After 12 overs: MI scored 95/2, runs_last_5=45
score, (low, high) = predict_live_score(
    'Mumbai Indians', 'Chennai Super Kings', 'Wankhede Stadium',
    12.0, 95, 2, 45, 0
)
print(f"Predicted Score: {score} (Range: {low}-{high})")
# Output: Predicted Score: 185 (Range: 165-205)
```

**Q5: What challenges exist in cricket score prediction?**

**A5:**

**1. High Variance:**
- Individual brilliance (one player scores 100)
- Match-winning partnerships
- Collapses (5 wickets in 10 runs)

**2. Contextual Factors:**
- Pitch conditions (batting-friendly vs bowling-friendly)
- Weather (dew helps batting in second innings)
- Pressure situations (finals vs league matches)
- Toss decision impact

**3. Limited Historical Data:**
- Only ~800 IPL matches till date
- Teams, players change each season
- Venue characteristics evolve

**4. Non-Stationary:**
- Game evolving (T20 tactics changing)
- Player form fluctuates
- Rule changes affect scoring

**Solutions:**
```python
# 1. Include more contextual features
df['is_playoff'] = df['match_type'].apply(lambda x: 1 if x == 'Playoff' else 0)
df['toss_decision'] = df['toss_decision'].apply(lambda x: 1 if x == 'bat' else 0)

# 2. Player-level features
df['strike_rate_batsman'] = df['batsman_runs'] / df['balls_faced']
df['economy_bowler'] = df['bowler_runs'] / df['bowler_overs']

# 3. Rolling averages
df['team_avg_last_5'] = df.groupby('batting_team')['total'].transform(
    lambda x: x.rolling(5, min_periods=1).mean()
)

# 4. Ensemble methods
ensemble = VotingRegressor([
    ('rf', RandomForestRegressor(100)),
    ('xgb', XGBRegressor(100)),
    ('lgb', LGBMRegressor(100))
])
```

---

## Additional Resources

**Cricket Analytics:**
- ESPNCricinfo Stats: Historical match data
- "The Art of Cricket Analytics" - Articles and papers

**Machine Learning:**
- Time series feature engineering
- Sports analytics with ML

