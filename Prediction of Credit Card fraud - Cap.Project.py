#!/usr/bin/env python
# coding: utf-8

# # Importing basic modules

# In[4]:


import pandas as pd

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Display the first few rows of the dataset
print(data.head())


# In[5]:


# Display summary statistics
print(data.describe())

# Display class distribution
print(data['Class'].value_counts())


# # Data Quality Check

# In[3]:


# Check for missing values
print(data.isnull().sum())

# Outlier detection can be done using various methods such as IQR or Z-score
# For example, using IQR for 'Amount' column

Q1 = data['Amount'].quantile(0.25)
Q3 = data['Amount'].quantile(0.75)
IQR = Q3 - Q1

outliers = data[(data['Amount'] < (Q1 - 1.5 * IQR)) | (data['Amount'] > (Q3 + 1.5 * IQR))]
print(f'Number of outliers in Amount: {len(outliers)}')


# # Data Visualization

# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns

# Visualize distribution of 'Time'
plt.figure(figsize=(10,6))
sns.histplot(data['Time'], bins=50)
plt.title('Distribution of Transactions Over Time')
plt.xlabel('Time')
plt.ylabel('Number of Transactions')
plt.show()

# Visualize distribution of 'Amount'
plt.figure(figsize=(10,6))
sns.histplot(data['Amount'], bins=50)
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Amount')
plt.ylabel('Number of Transactions')
plt.show()

# Correlation matrix
plt.figure(figsize=(20,10))
sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# # Data Cleaning

# In[13]:


# Convert 'Time' column to datetime format
data['Time'] = pd.to_datetime(data['Time'], unit='s')


# # Dealing with Imbalanced Data

# In[7]:


# Separate features and target variable
X = data.drop('Class', axis=1)
y = data['Class']


# In[8]:


y.value_counts()


# In[10]:


from imblearn.over_sampling import SMOTE

# Separate features and target variable
X = data.drop('Class', axis=1)
y = data['Class']

# Apply SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Check the new class distribution
print(pd.Series(y_res).value_counts())


# In[9]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# # Feature Engineering

# In[11]:


from sklearn.preprocessing import StandardScaler

# Scale the 'Amount' feature
scaler = StandardScaler()
data['Amount_scaled'] = scaler.fit_transform(data[['Amount']])


# # Model Selection and Training

# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# In[16]:


# Initialize models
lr = LogisticRegression()
rf = RandomForestClassifier()
xgb = XGBClassifier()


# In[17]:


# Train models
lr.fit(X_train, y_train)


# In[21]:


rf.fit(X_train, y_train)


# In[19]:


xgb.fit(X_train, y_train)


# Model Validation

# In[23]:


from sklearn.metrics import classification_report, roc_auc_score

# Predictions
y_pred_lr = lr.predict(X_test)

y_pred_xgb = xgb.predict(X_test)

# Classification reports
print("Logistic Regression Report")
print(classification_report(y_test, y_pred_lr))
print("XGBoost Report")
print(classification_report(y_test, y_pred_xgb))

# ROC-AUC Scores
print(f"Logistic Regression ROC-AUC: {roc_auc_score(y_test, y_pred_lr)}")
print(f"XGBoost ROC-AUC: {roc_auc_score(y_test, y_pred_xgb)}")

