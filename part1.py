
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

## EDA

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
