#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


# Load the dataset
train_data = pd.read_csv("playground-series-s4e7/train.csv")


# In[7]:


# Display the first few rows of the dataset
print(train_data.head())


# In[8]:


# Check the shape of the dataset
print(f"Dataset shape: {train_data.shape}")


# In[9]:


# Check the data types of each column
print(train_data.dtypes)


# In[10]:


# Check for unique values in the 'id' column
print(f"Number of unique IDs: {train_data.id.nunique()}")


# In[11]:


# Drop the 'id' column as it is not useful for the model
train_data.drop(columns=['id'], inplace=True)


# In[12]:


# Check for missing values
print(f"Missing values:\n{train_data.isnull().mean()}")


# In[13]:


# Display unique values for each column
for var in train_data.columns:
    print(f"{var}: {train_data[var].unique()[:20]} | {train_data[var].nunique()}\n")


# In[14]:


continuous = list(train_data.select_dtypes(exclude="O").columns)[:-1]
categorical = list(train_data.select_dtypes(include="O").columns)

print(f"There are {len(continuous)} continuous variables: {continuous}")
print(f"There are {len(categorical)} categorical variables: {categorical}")

# Check the number of unique values in categorical variables
print(train_data[categorical].nunique())


# In[15]:


num_cols = len(continuous)
fig, axes = plt.subplots(num_cols, 1, figsize=(10, 4 * num_cols))

for i, col in enumerate(continuous):
    train_data.boxplot(column=col, ax=axes[i])
    axes[i].set_title(f'Boxplot of {col}')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Values')

plt.tight_layout()
plt.show()


# In[16]:


train_data[continuous].hist(bins=30, figsize=(15, 15))
plt.show()


# In[17]:


from sklearn.model_selection import train_test_split

X = train_data.drop('Response', axis=1)
y = train_data['Response']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")


# In[18]:


from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score


# In[19]:


catboost_classifier = CatBoostClassifier(random_seed=0, verbose=100)


# In[20]:


catboost_classifier.fit(X_train, y_train, cat_features=categorical)


# In[21]:


X_train_preds = catboost_classifier.predict_proba(X_train)[:, 1]
X_test_preds = catboost_classifier.predict_proba(X_test)[:, 1]


# In[22]:


print('Train set')
print('xgb roc-auc: {:.4f}'.format(roc_auc_score(y_train, X_train_preds)))

print('Test set')
print('xgb roc-auc: {:.4f}'.format(roc_auc_score(y_test, X_test_preds)))


# In[23]:


# Continue training
catboost_classifier.fit(X_train, y_train, cat_features=categorical, init_model=catboost_classifier)


# In[24]:


X_train_preds = catboost_classifier.predict_proba(X_train)[:, 1]
X_test_preds = catboost_classifier.predict_proba(X_test)[:, 1]


# In[25]:


print('Continued Train set')
print('catboost roc-auc: {:.4f}'.format(roc_auc_score(y_train, X_train_preds)))

print('Continued Test set')
print('catboost roc-auc: {:.4f}'.format(roc_auc_score(y_test, X_test_preds)))


# In[26]:


importance = pd.Series(catboost_classifier.feature_importances_, index=X.columns)
importance.sort_values(ascending=False).plot.bar(figsize=(12,6))
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()


# In[27]:


test_data = pd.read_csv("playground-series-s4e7/test.csv")


# In[28]:


test_ids = test_data['id']
test_data.drop(columns = ["id"], inplace = True)


# In[29]:


mappings_list = [{'Male': 0, 'Female': 1}, {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}, {'Yes': 1, 'No': 0}]
test_data["Gender"] = test_data["Gender"].map(mappings_list[0])
test_data["Vehicle_Age"] = test_data["Vehicle_Age"].map(mappings_list[1])
test_data["Vehicle_Damage"] = test_data["Vehicle_Damage"].map(mappings_list[2])


# In[30]:


predictions_test = catboost_classifier.predict_proba(test_data)[:, 1]


# In[31]:


result = pd.DataFrame({'id': test_ids, 'Response': predictions_test.flatten()}, columns=['id', 'Response'])
result.to_csv("playground-series-s4e7/submission.csv", index=False)


# In[ ]:


# ##########################
#stack of xgboost and catboost

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# Load the training and test datasets
train_data = pd.read_csv("playground-series-s4e7/train.csv")
test_data = pd.read_csv("playground-series-s4e7/test.csv")

# Drop the 'id' column as it is not useful for the model
train_data.drop(columns=['id'], inplace=True)

# Separate features and target variable from the training dataset
X_train = train_data.drop('Response', axis=1)
y_train = train_data['Response']

# Load and preprocess the test dataset
test_ids = test_data['id']
test_data.drop(columns=['id'], inplace=True)

# Map categorical variables in the test dataset
mappings_list = [{'Male': 0, 'Female': 1}, {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}, {'Yes': 1, 'No': 0}]
X_train["Gender"] = X_train["Gender"].map(mappings_list[0])
X_train["Vehicle_Age"] = X_train["Vehicle_Age"].map(mappings_list[1])
X_train["Vehicle_Damage"] = X_train["Vehicle_Damage"].map(mappings_list[2])

test_data["Gender"] = test_data["Gender"].map(mappings_list[0])
test_data["Vehicle_Age"] = test_data["Vehicle_Age"].map(mappings_list[1])
test_data["Vehicle_Damage"] = test_data["Vehicle_Damage"].map(mappings_list[2])

# Identify categorical features
categorical = list(X_train.select_dtypes(include="O").columns)

# Train the CatBoost model
catboost_classifier = CatBoostClassifier(random_seed=0, verbose=100)
catboost_classifier.fit(X_train, y_train, cat_features=categorical)

# Train the XGBoost model
xgboost_classifier = XGBClassifier(random_state=0, use_label_encoder=False)
xgboost_classifier.fit(X_train, y_train)

# Make predictions on the test dataset using CatBoost
predictions_test_catboost = catboost_classifier.predict_proba(test_data)[:, 1]

# Make predictions on the test dataset using XGBoost
predictions_test_xgboost = xgboost_classifier.predict_proba(test_data)[:, 1]

# Ensemble predictions using a weighted average
ensemble_predictions = (0.5 * predictions_test_catboost) + (0.5 * predictions_test_xgboost)

# Create a submission file
result = pd.DataFrame({'id': test_ids, 'Response': ensemble_predictions.flatten()}, columns=['id', 'Response'])
result.to_csv("playground-series-s4e7/submission.csv", index=False)

# Make predictions on the training dataset using CatBoost
X_train_preds_catboost = catboost_classifier.predict_proba(X_train)[:, 1]

# Make predictions on the training dataset using XGBoost
X_train_preds_xgboost = xgboost_classifier.predict_proba(X_train)[:, 1]

# Ensemble predictions on the training dataset using a weighted average
X_train_preds_ensemble = (0.5 * X_train_preds_catboost) + (0.5 * X_train_preds_xgboost)

# Calculate and display the ROC AUC score for the training set (CatBoost)
print('Train set (CatBoost)')
print('catboost roc-auc: {:.4f}'.format(roc_auc_score(y_train, X_train_preds_catboost)))

# Calculate and display the ROC AUC score for the training set (XGBoost)
print('Train set (XGBoost)')
print('xgboost roc-auc: {:.4f}'.format(roc_auc_score(y_train, X_train_preds_xgboost)))

# Calculate and display the ROC AUC score for the training set (Ensemble)
print('Train set (Ensemble)')
print('ensemble roc-auc: {:.4f}'.format(roc_auc_score(y_train, X_train_preds_ensemble)))

# Plot feature importance for CatBoost
importance_catboost = pd.Series(catboost_classifier.feature_importances_, index=X_train.columns)
importance_catboost.sort_values(ascending=False).plot.bar(figsize=(12, 6))
plt.title('Feature Importance (CatBoost)')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

# Plot feature importance for XGBoost
importance_xgboost = pd.Series(xgboost_classifier.feature_importances_, index=X_train.columns)
importance_xgboost.sort_values(ascending=False).plot.bar(figsize=(12, 6))
plt.title('Feature Importance (XGBoost)')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

