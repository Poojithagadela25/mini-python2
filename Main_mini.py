import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv(r"C:\Users\pooji\kdrama_DATASET.csv")

# Initial exploration
print('\n--- DataFrame Head ---')
print(df.head())

df.columns = df.columns.str.strip()
print('\n--- Columns ---')
print(df.columns)

print('\n--- Missing Values ---')
print(df.isnull().sum())

print('\n--- Stats Summary ---')
print(df.describe())

# Drop rows with missing key values
df_clean = df.dropna(subset=['Rating', 'Number of Episodes', 'Year of release'])

# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Number of Episodes', y='Rating', data=df_clean)
plt.title('Box Plot: Rating Distribution by Number of Episodes')
plt.xlabel('Number of Episodes')
plt.ylabel('Rating')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Scatterplot
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_clean, x='Year of release', y='Rating',
                hue='Number of Episodes', size='Number of Episodes',
                palette='viridis', legend='brief')
plt.title('Scatter Plot: Ratings Over Years')
plt.xlabel('Year of Release')
plt.ylabel('Rating')
plt.legend(title='Episode Count', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(df_clean[['Rating', 'Number of Episodes', 'Year of release']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Data cleaning and conversion
df_clean['Rating'] = pd.to_numeric(df_clean['Rating'], errors='coerce')
df_clean['Number of Episodes'] = pd.to_numeric(df_clean['Number of Episodes'], errors='coerce')
df_clean['Year of release'] = pd.to_numeric(df_clean['Year of release'], errors='coerce')

# Add High Rating flag
df_clean['High Rating'] = (df_clean['Rating'] >= 9.0).astype(int)

# Linear Regression
x = df_clean[['Year of release', 'Number of Episodes']]
y = df_clean['Rating'].astype(int)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

lin_model = LinearRegression()
lin_model.fit(x_train, y_train)
y_pred = lin_model.predict(x_test)

print(f'Linear Regression Accuracy: {lin_model.score(x_test, y_test):.2f}')
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f'MSE: {mse:.3f}, MAE: {mae:.3f}, MAPE: {mape:.3f}%')

# Logistic Regression
x_log = df_clean[['Year of release', 'Number of Episodes']]
y_log = df_clean['High Rating'].astype(int)
x_train_log, x_test_log, y_train_log, y_test_log = train_test_split(x_log, y_log, test_size=0.2, random_state=42)

log_model = LogisticRegression(class_weight='balanced')
log_model.fit(x_train_log, y_train_log)
y_pred_log = log_model.predict(x_test_log)

print(f'Logistic Regression Accuracy: {log_model.score(x_test_log, y_test_log):.2f}')
print(f'Precision: {precision_score(y_test_log, y_pred_log, zero_division=0):.3f}')
print(f'Recall: {recall_score(y_test_log, y_pred_log, zero_division=0):.3f}')
print(f'F1 Score: {f1_score(y_test_log, y_pred_log, zero_division=0):.3f}')

# Decision Tree
x_tree = df_clean[['Year of release', 'Number of Episodes']]
y_tree = df_clean['High Rating']
x_train_tree, x_test_tree, y_train_tree, y_test_tree = train_test_split(x_tree, y_tree, test_size=0.2, random_state=42)

tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_model.fit(x_train_tree, y_train_tree)
y_pred_tree = tree_model.predict(x_test_tree)

tree_accuracy = accuracy_score(y_test_tree, y_pred_tree)
print(f'Decision Tree Accuracy: {tree_accuracy:.2f}')

plt.figure(figsize=(12, 6))
plot_tree(tree_model, feature_names=x_tree.columns, class_names=['Low', 'High'], filled=True)
plt.title('Decision Tree: Predicting High Rating')
plt.show()

# Gradient Boosting
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(x_train, y_train)
y_pred_gb = gb_model.predict(x_test)

y_test = y_test.astype(int)
y_pred_gb = y_pred_gb.astype(int)

print('Gradient Boosting Classifier')
print(f'Accuracy: {accuracy_score(y_test, y_pred_gb)}')