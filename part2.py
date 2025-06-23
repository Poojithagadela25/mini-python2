
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/Users/nawazshareefshaik/Documents/PYTHON/Midterm/kdrama_DATASET.csv")
df.head()
df_clean = df.dropna(subset=['Rating', 'Number of Episodes', 'Year of release'])
df_clean

# # Feature Engineering

df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df['Number of Episodes'] = pd.to_numeric(df['Number of Episodes'], errors='coerce')
df['Year of release'] = pd.to_numeric(df['Year of release'], errors='coerce')
df_clean['High Rating'] = (df_clean['Rating'] >= 9.0).astype(int)

# # Linear Regression

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
X = df_clean[['Year of release', 'Number of Episodes']]
y = df_clean['Rating'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred = lin_model.predict(X_test)
print(f"Linear Regression Accuracy: {lin_model.score(X_test, y_test):.2f}")
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mase = mae / np.mean(np.abs(y_test - y_test.mean()))

print(f"MSE: {mse:.3f}, MAE: {mae:.3f}, MASE: {mase:.3f}")

# # Logistic Regression

df_clean['High Rating'] = (df_clean['Rating'] >= 9.0).astype(int)
X_log = df_clean[['Year of release', 'Number of Episodes']]
y_log = df_clean['High Rating']
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=0.2, random_state=42)
log_model = LogisticRegression(class_weight='balanced')
log_model.fit(X_train_log, y_train_log)
print(f"Logistic Regression Accuracy: {log_model.score(X_test_log, y_test_log):.2f}")
y_pred_log = log_model.predict(X_test_log)
from sklearn.metrics import precision_score, recall_score, f1_score
precision = precision_score(y_test_log, y_pred_log, zero_division=0)
recall = recall_score(y_test_log, y_pred_log, zero_division=0)
f1 = f1_score(y_test_log, y_pred_log, zero_division=0)
print(f"Logistic Regression Accuracy: {log_model.score(X_test_log, y_test_log):.2f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")

