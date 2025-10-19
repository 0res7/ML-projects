# Interview Preparation: Time Series Multi-Store Sales Prediction

## 1. Project Overview

**Problem Statement:** Predict daily sales for every product and store combination for the next month using historical sales data from a large Russian software firm (1C Company).

**Objective:** Build time series forecasting models with lag features to predict future sales, enabling inventory optimization and demand planning.

**Challenge:** Large-scale prediction (thousands of product-store combinations), time series features, seasonal patterns.

---

## 2. Technical Concepts

### Time Series Forecasting
- **Temporal Dependencies:** Past sales influence future
- **Lag Features:** Use shifted values as predictors
- **Seasonality:** Weekly, monthly, yearly patterns
- **Trend:** Long-term increasing/decreasing pattern

### Algorithms
- **Linear Regression with Lag Features**
- **XGBoost Regressor:** Gradient boosting for time series
- **LightGBM:** Fast gradient boosting
- **ARIMA/SARIMA:** Classical time series (optional)
- **LSTM:** Deep learning for sequences (advanced)

---

## 3. Mathematical Foundations

### Lag Features
For time series \(y_t\):
\[
y_t = f(y_{t-1}, y_{t-2}, ..., y_{t-k})
\]

**Example:**
```
Day 10 sales = f(Day 9 sales, Day 8 sales, Day 7 sales)
```

### Auto-Regression (AR)
\[
y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + ... + \phi_p y_{t-p} + \epsilon_t
\]

### Moving Average (MA)
\[
y_t = \mu + \epsilon_t + \theta_1\epsilon_{t-1} + ... + \theta_q\epsilon_{t-q}
\]

### Root Mean Squared Error (Competition Metric)
\[
RMSE = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}
\]
Target values clipped to [0, 20] range.

---

## 4. Implementation Details

### Data Structure
```python
# sales_train.csv
columns = ['date', 'date_block_num', 'shop_id', 'item_id', 
          'item_price', 'item_cnt_day']

# shop_id: Store identifier
# item_id: Product identifier  
# date_block_num: Month number (0-33)
# item_cnt_day: Daily sales (target)
```

### Feature Engineering

**1. Lag Features:**
```python
# Previous month sales
df['lag_1'] = df.groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(1)
df['lag_2'] = df.groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(2)
df['lag_3'] = df.groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(3)
df['lag_6'] = df.groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(6)
df['lag_12'] = df.groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(12)
```

**2. Rolling Statistics:**
```python
# Moving averages
df['rolling_mean_3'] = df.groupby(['shop_id', 'item_id'])['item_cnt_month'].transform(
    lambda x: x.rolling(window=3).mean()
)
df['rolling_mean_6'] = df.groupby(['shop_id', 'item_id'])['item_cnt_month'].transform(
    lambda x: x.rolling(window=6).mean()
)

# Rolling std (volatility)
df['rolling_std_3'] = df.groupby(['shop_id', 'item_id'])['item_cnt_month'].transform(
    lambda x: x.rolling(window=3).std()
)
```

**3. Trend Features:**
```python
# Month of year (seasonality)
df['month'] = df['date_block_num'] % 12

# Overall trend
df['trend'] = df['date_block_num']
```

**4. Aggregated Features:**
```python
# Shop-level monthly sales
shop_monthly = df.groupby(['shop_id', 'date_block_num'])['item_cnt_day'].sum().reset_index()
shop_monthly.rename(columns={'item_cnt_day': 'shop_monthly_sales'}, inplace=True)
df = df.merge(shop_monthly, on=['shop_id', 'date_block_num'])

# Item-level monthly sales
item_monthly = df.groupby(['item_id', 'date_block_num'])['item_cnt_day'].sum().reset_index()
item_monthly.rename(columns={'item_cnt_day': 'item_monthly_sales'}, inplace=True)
df = df.merge(item_monthly, on=['item_id', 'date_block_num'])
```

### Workflow
```python
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load data
sales_train = pd.read_csv('sales_train.csv')
items = pd.read_csv('items.csv')
shops = pd.read_csv('shops.csv')

# Aggregate to monthly level
monthly = sales_train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({
    'item_cnt_day': 'sum',
    'item_price': 'mean'
}).reset_index()
monthly.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace=True)

# Clip target (competition rule)
monthly['item_cnt_month'] = monthly['item_cnt_month'].clip(0, 20)

# Create lag features
for lag in [1, 2, 3, 6, 12]:
    monthly[f'lag_{lag}'] = monthly.groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(lag)

# Drop rows with NaN (first 12 months lose to lag features)
monthly = monthly.dropna()

# Train-test split (temporal)
# Train: months 0-32, Test: month 33
train = monthly[monthly['date_block_num'] < 33]
test = monthly[monthly['date_block_num'] == 33]

features = ['shop_id', 'item_id', 'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12']
X_train = train[features]
y_train = train['item_cnt_month']
X_test = test[features]
y_test = test['item_cnt_month']

# Train XGBoost
xgb = XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb.fit(X_train, y_train)

# Predict
y_pred = xgb.predict(X_test).clip(0, 20)  # Clip predictions

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")
```

---

## 5. Outcomes & Results

### Typical Performance
- **RMSE:** 0.90-1.10 (Kaggle competition)
- **Baseline (Mean):** ~1.7 RMSE
- **Linear Regression:** ~1.06 RMSE
- **XGBoost:** ~0.94 RMSE

### Key Insights
- **Lag Features Critical:** lag_1 most important
- **Seasonality:** December sales spike (holidays)
- **Cold Start:** New products hard to predict (no history)
- **Sparsity:** Many shop-item combinations have zero sales

---

## 6. Interview Questions & Answers

**Q1: Why is temporal train-test split important?**

**A1:** **Prevent Data Leakage**

**Wrong (Random Split):**
```python
# BAD: Random split mixes past and future
train_test_split(X, y, test_size=0.2)  
# Model sees future data during training!
```

**Correct (Temporal Split):**
```python
# Use chronological split
train = df[df['date_block_num'] < 33]
test = df[df['date_block_num'] == 33]
```

**Realistic Evaluation:**
- Mirrors production (predicting future from past)
- Tests model's ability to generalize forward in time

**Q2: What are lag features and why are they important?**

**A2:** **Lag Features: Shifted Time Series Values**

```python
# Original
date_block_num | item_cnt_month
0              | 10
1              | 12
2              | 15
3              | 18

# With lag_1 (previous month)
date_block_num | item_cnt_month | lag_1
1              | 12             | 10
2              | 15             | 12
3              | 18             | 15
```

**Why Important:**
- **Temporal Dependency:** Past sales predict future
- **Trend Capture:** Increasing/decreasing patterns
- **Seasonality:** Annual/monthly patterns via lag_12

**Q3: How do you handle new products with no sales history?**

**A3:** **Cold Start Problem**

**Solutions:**
```python
# 1. Use item category average
new_product_pred = category_avg_sales[item_category]

# 2. Similar product matching
similar_items = find_similar_items(new_item, item_features)
new_product_pred = similar_items_sales.mean()

# 3. Global average (worst case)
new_product_pred = overall_avg_sales

# 4. Use product attributes
# Price, category, brand → regression model
```

**Q4: Why clip predictions to [0, 20]?**

**A4:** **Competition Rule + Realistic Bounds**

- **Lower Bound (0):** Can't have negative sales
- **Upper Bound (20):** Monthly sales rarely exceed 20 for single product-store
- **Outlier Handling:** Prevents extreme predictions from affecting RMSE
- **Domain Knowledge:** Retail context has natural limits

**Q5: How would you improve this model for production?**

**A5:**

**1. More Lag Features:**
```python
# Different window sizes
lags = [1, 2, 3, 4, 5, 6, 12, 24]
for lag in lags:
    df[f'lag_{lag}'] = df.groupby(['shop_id', 'item_id'])['sales'].shift(lag)
```

**2. External Features:**
- Holidays, promotions, weather
- Competitor pricing
- Economic indicators

**3. Hierarchical Models:**
```python
# Level 1: Predict category sales
# Level 2: Distribute to items within category
```

**4. Deep Learning:**
```python
from keras.layers import LSTM, Dense
# LSTM for sequence modeling
```

**5. Ensemble:**
```python
# Combine multiple approaches
final_pred = 0.4 * xgb_pred + 0.3 * lgb_pred + 0.3 * lstm_pred
```

---

## Additional Resources

**Competitions:**
- Kaggle: "Predict Future Sales" (1C Company)
- M5 Forecasting (Walmart)

**Papers:**
- "Time Series Forecasting with XGBoost" - Practitioners Guide
- Hyndman & Athanasopoulos: "Forecasting: Principles and Practice"

