
# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[23]:


df = pd.read_csv("/Users/nawazshareefshaik/Documents/PYTHON/Midterm/kdrama_DATASET.csv")
df.head()


# In[24]:


df_clean = df.dropna(subset=['Rating', 'Number of Episodes', 'Year of release'])
df_clean


# # Feature Engineering

# In[25]:


df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df['Number of Episodes'] = pd.to_numeric(df['Number of Episodes'], errors='coerce')
df['Year of release'] = pd.to_numeric(df['Year of release'], errors='coerce')


# In[26]:


df_clean['High Rating'] = (df_clean['Rating'] >= 9.0).astype(int)


# # Linear Regression

# In[27]:


from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


# In[28]:


X = df_clean[['Year of release', 'Number of Episodes']]
y = df_clean['Rating'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[29]:


lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred = lin_model.predict(X_test)


# In[30]:


print(f"Linear Regression Accuracy: {lin_model.score(X_test, y_test):.2f}")


# In[31]:


mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mase = mae / np.mean(np.abs(y_test - y_test.mean()))

print(f"MSE: {mse:.3f}, MAE: {mae:.3f}, MASE: {mase:.3f}")


# # Logistic Regression

# In[32]:


df_clean['High Rating'] = (df_clean['Rating'] >= 9.0).astype(int)
X_log = df_clean[['Year of release', 'Number of Episodes']]
y_log = df_clean['High Rating']
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=0.2, random_state=42)


# In[33]:


log_model = LogisticRegression(class_weight='balanced')
log_model.fit(X_train_log, y_train_log)
print(f"Logistic Regression Accuracy: {log_model.score(X_test_log, y_test_log):.2f}")


# In[34]:


y_pred_log = log_model.predict(X_test_log)


# In[35]:


from sklearn.metrics import precision_score, recall_score, f1_score


# In[36]:


precision = precision_score(y_test_log, y_pred_log, zero_division=0)
recall = recall_score(y_test_log, y_pred_log, zero_division=0)
f1 = f1_score(y_test_log, y_pred_log, zero_division=0)


# In[37]:


print(f"Logistic Regression Accuracy: {log_model.score(X_test_log, y_test_log):.2f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")

