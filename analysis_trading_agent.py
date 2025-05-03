#%% Importing libraries and data

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('final_merged_features.csv')

# Now manually convert index
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # if needed
df = df.set_index('Date')


#%% XGBoost modelling

#  Target Engineering
# ADM_Return > 0 → Long (1), ADM_Return <= 0 → Short (-1)
df['Target'] = np.where(df['ADM_y'] > 0, 1, -1)


#  Define Features and Target
X = df.drop(columns=['ADM_x', 'ADM_y', 'Target'])
y = df['Target']

#  Train/Test Split (TimeSeriesSplit or simple chronological split)
split_date = '2020-01-01'  # you can adjust
X_train = X.loc[X.index < split_date]
X_test = X.loc[X.index >= split_date]
y_train = y.loc[y.index < split_date]
y_test = y.loc[y.index >= split_date]

#  Remap -1 → 0
y_train = y_train.replace(-1, 0)
y_test = y_test.replace(-1, 0)

#  Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  Train XGBoost Classifier
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train_scaled, y_train)

#  Make Predictions
preds = model.predict(X_test_scaled)

#%% Model results and diagnostic

#  Accuracy

from sklearn.metrics import accuracy_score
print(f"Classification Accuracy: {accuracy_score(y_test, preds):.4f}")

preds = np.where(preds == 0, -1, 1)

# Evaluate
# Trading returns: if correct prediction, earn ADM return; else lose ADM return
test_returns = df.loc[X_test.index, 'ADM_y']

# Trading strategy returns
strategy_returns = preds * test_returns

# Cumulative returns
cumulative_strategy = (1 + strategy_returns).cumprod()
cumulative_buy_hold = (1 + test_returns).cumprod()

# Plot
plt.figure(figsize=(12,6))
plt.plot(cumulative_strategy, label='Strategy (ML Model)')
plt.plot(cumulative_buy_hold, label='Buy and Hold ADM')
plt.title('ML Trading Strategy vs Buy & Hold')
plt.legend()
plt.grid()
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Plot confusion matrix
cm = confusion_matrix(y_test, (preds == 1).astype(int))  # Convert back to 0/1 for matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Short (0)', 'Long (1)'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.grid(False)
plt.show()


backtest = pd.DataFrame(strategy_returns)
backtest = backtest.rename(columns={'ADM_y': 'str_return'})
backtest.to_excel("backtest.xlsx")
