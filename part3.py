
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("kdrama_DATASET.csv")
df.head()

df.columns = df.columns.str.strip()
df.columns

df.isnull().sum()

df.describe()

# # EDA

df_clean = df.dropna(subset=['Rating', 'Number of Episodes', 'Year of release'])
df_clean

# # Visualisation

plt.figure(figsize=(10, 6))
sns.boxplot(x='Number of Episodes', y='Rating', data=df_clean)
plt.title("Box Plot: Rating Distribution by Number of Episodes")
plt.xlabel("Number of Episodes")
plt.ylabel("Rating")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_clean, x='Year of release', y='Rating',
                hue='Number of Episodes', size='Number of Episodes',
                palette='viridis', legend='brief')
plt.title("Scatter Plot: Ratings Over Years")
plt.xlabel("Year of Release")
plt.ylabel("Rating")
plt.legend(title='Episode Count', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_clean, x='Year of release', y='Rating',
                hue='Number of Episodes', size='Number of Episodes',
                palette='viridis', legend='brief')
plt.title("Scatter Plot: Ratings Over Years")
plt.xlabel("Year of Release")
plt.ylabel("Rating")
plt.legend(title='Episode Count', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

plt.figure(figsize=(4, 2))
sns.heatmap(df_clean[['Rating', 'Number of Episodes', 'Year of release']].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# # Feature Engineering

df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df['Number of Episodes'] = pd.to_numeric(df['Number of Episodes'], errors='coerce')
df['Year of release'] = pd.to_numeric(df['Year of release'], errors='coerce')

df_clean['High Rating'] = (df_clean['Rating'] >= 9.0).astype(int)

# # Decision Tree

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

df_clean['High Rating'] = (df_clean['Rating'] >= 9.0).astype(int)
X_tree = df_clean[['Year of release', 'Number of Episodes']]
y_tree = df_clean['High Rating']

X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(X_tree, y_tree, test_size=0.2, random_state=42)

tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_model.fit(X_train_tree, y_train_tree)
y_pred_tree = tree_model.predict(X_test_tree)

tree_model

# Accuracy
tree_accuracy = accuracy_score(y_test_tree, y_pred_tree)
print(f"Decision Tree Accuracy: {tree_accuracy:.2f}")

plt.figure(figsize=(12, 6))
plot_tree(tree_model, feature_names=X_tree.columns, class_names=['Low', 'High'], filled=True)
plt.title("Decision Tree: Predicting High Rating")
plt.show()

## Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

X = df_clean[['Year of release', 'Number of Episodes']]
y = df_clean['Rating'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

gb_model

y_test = y_test.astype(int)
y_pred_gb = y_pred_gb.astype(int)

y_pred_gb

print("Gradient Boosting Classifier")
print("Accuracy:", accuracy_score(y_test, y_pred_gb))
