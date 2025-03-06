#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# In[62]:


df = pd.read_csv("./iot_dataset.csv")


# In[63]:


df.head(10)


# In[64]:


df.tail()


# In[65]:


df.info()


# In[66]:


df.shape


# In[67]:


df.isnull().sum()


# In[68]:


df.drop(columns=['created_at', 'entry_id'], inplace=True)


# In[69]:


df.head()


# In[70]:


df.tail()


# In[71]:


def classify_fault(row):

    conditions = [
        {
            "fault_type": "Overheating and Overvoltage",
            "cause": "High voltage causing excessive heat or cooling system failure",
            "condition": row["voltage"] > 240 and row["temp"] > 90
        },
        {
            "fault_type": "Overvoltage",
            "cause": "Grid surge or faulty voltage regulator",
            "condition": row["voltage"] > 240
        },
        {
            "fault_type": "Undervoltage",
            "cause": "Grid instability or transformer overload",
            "condition": row["voltage"] < 180
        },
        {
            "fault_type": "Normal Condition",
            "cause": "All parameters within normal operating range",
            "condition": True  # Default case
        }
    ]


    for condition in conditions:
        if condition["condition"]:
            return condition["fault_type"],condition["cause"]#,condition["cause"]


# In[12]:


# def classify_fault(row):
#     conditions = [
#         {
#             "fault_type": "Overheating and Overvoltage",
#             "cause": "High voltage causing excessive heat or cooling system failure",
#             "condition": row["voltage"] > 240 and row["temp"] > 90
#         },
#         {
#             "fault_type": "Overheating",
#             "cause": "Cooling system failure or high ambient temperature",
#             "condition": row["temp"] > 90
#         },
#         {
#             "fault_type": "Overvoltage",
#             "cause": "Grid surge or faulty voltage regulator",
#             "condition": row["voltage"] > 240
#         },
#         {
#             "fault_type": "Undervoltage",
#             "cause": "Grid instability or transformer overload",
#             "condition": row["voltage"] < 180
#         },
#         {
#             "fault_type": "Overheating and Undervoltage",
#             "cause": "Cooling system failure combined with low grid voltage",
#             "condition": row["voltage"] < 180 and row["temp"] > 90
#         },
#         {
#             "fault_type": "Normal Condition",
#             "cause": "All parameters within normal operating range",
#             "condition": True  # Default case
#         }
#     ]

#     # Iterate and return the first matching condition
#     for condition in conditions:
#         if condition["condition"]:
#             return condition["fault_type"], condition["cause"]


# In[72]:


df.head()


# In[73]:


# Apply fault classification
df["Fault_Type_and_Cause"] = df.apply(classify_fault, axis=1)
df[["Fault_Type", "Cause"]] = df["Fault_Type_and_Cause"].apply(pd.Series)
df.drop(columns=["Fault_Type_and_Cause"], inplace=True)


# In[74]:


df.head()


# In[75]:


df.tail()


# In[76]:


df.shape


# In[77]:


df.info


# In[78]:


df.isnull().sum()


# In[79]:


df[["Fault_Type", "Cause"]].value_counts()


# In[80]:


X = df.drop(columns=["Fault_Type", "Cause"])
y = df["Fault_Type"]


# In[81]:


print(X)


# In[82]:


print(y)


# In[83]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[84]:


# Label Encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# In[85]:


label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Encoding Mapping:", label_mapping)


# In[86]:


# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[87]:


# Convert scaled data back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)


# In[88]:


X_train.shape, X_test.shape


# In[89]:


scaler.mean_


# In[90]:


X_train_scaled # transformed scaled value by Dataframe


# In[91]:


np.round(X_train.describe(),1)


# In[92]:


np.round(X_train_scaled.describe(),1)


# In[93]:


# RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)

# Fit the model on training data
rf_classifier.fit(X_train, y_train)


# In[95]:


# Predict on test data
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)


# In[96]:


#GradientBoostingClassifier

from sklearn.ensemble import GradientBoostingClassifier

# Initialize the Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(random_state=42)

# Fit the model on training data
gb_classifier.fit(X_train, y_train)


# In[39]:


# Predict on test data
y_pred = gb_classifier.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print accuracy
print("Accuracy Score:", accuracy)


# In[40]:


#Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

# Initialize the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Fit the model on training data
dt_classifier.fit(X_train, y_train)


# In[41]:


# Predict on test data
y_pred = dt_classifier.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print accuracy
print("Accuracy Score:", accuracy)


# In[42]:


get_ipython().system('pip install xgboost')


# In[43]:


#XGBoost Classifier

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Apply Label Encoding on target variable
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Initialize XGBoost
xgb_classifier = XGBClassifier(random_state=42, eval_metric='mlogloss')

# Train the model
xgb_classifier.fit(X_train, y_train_encoded)

# Predict on test data
y_pred = xgb_classifier.predict(X_test)

# Decode labels for readable results (optional)
y_pred_decoded = le.inverse_transform(y_pred)


# In[44]:


accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"XGBoost Accuracy: {accuracy:.2f}")


# In[45]:


#SVM (Support Vector Machine)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Initialize SVM with RBF kernel
svm_classifier = SVC(kernel='rbf', random_state=42)

# Train the model
svm_classifier.fit(X_train, y_train)

# Predict on test data
y_pred = svm_classifier.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Accuracy: {accuracy:.2f}")


# In[46]:


#KNN (K-Nearest Neighbor)

from sklearn.neighbors import KNeighborsClassifier

# Initialize KNN
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn_classifier.fit(X_train, y_train)

# Predict on test data
y_pred = knn_classifier.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Accuracy: {accuracy:.2f}")


# In[47]:


get_ipython().system('pip install lightgbm')


# In[48]:


#LightGBM

from lightgbm import LGBMClassifier

# Initialize LightGBM
lgbm_classifier = LGBMClassifier(random_state=42)

# Train the model
lgbm_classifier.fit(X_train, y_train)

# Predict on test data
y_pred = lgbm_classifier.predict(X_test)




# In[49]:


# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"LightGBM Accuracy: {accuracy:.2f}")


# In[50]:


#Accuracy Score

y_pred = rf_classifier.predict(X_test)    # Predict the labels for the test set
accuracy = accuracy_score(y_test, y_pred)   # Calculate the accuracy score


# In[51]:


print(y_pred)


# In[97]:


# Creating a dictionary to map Fault_Type to Cause
fault_cause_mapping = dict(df[["Fault_Type","Cause"]].drop_duplicates().values)


# In[98]:


print(fault_cause_mapping)


# In[101]:


# # def predict_fault_and_cause(input_data):

# #     input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
# #     predicted_fault = rf_classifier.predict(input_data_as_numpy_array)[0]

# #     predicted_cause = fault_cause_mapping.get(predicted_fault, "Unknown Cause")

# #     return predicted_fault, predicted_cause


# # def predict_fault_and_cause(input_data):
# #     input_df = pd.DataFrame([input_data], columns=X_train.columns)  # Match feature names
# #     predicted_fault = rf_classifier.predict(input_df)[0]  # Predict fault type
# #     predicted_cause = fault_cause_mapping.get(predicted_fault, "Unknown Cause")  # Get cause

# #     return predicted_fault, predicted_cause


# input_data = (175,0.04,11.0,15.025,50.19,0.77,35)


# predicted_fault, predicted_cause = predict_fault_and_cause(input_data)

# print(f"Predicted Fault: {predicted_fault}")
# print(f"Cause: {predicted_cause}")
# # Overheating and Overvoltage	High voltage causing excessive heat or cooling...


# In[102]:


def predict_fault_and_cause(input_data):
    
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

    
    predicted_fault = rf_classifier.predict(input_data_as_numpy_array)[0]

    
    predicted_cause = fault_cause_mapping.get(predicted_fault, "Unknown Cause")

    return predicted_fault, predicted_cause



input_data = (150, 0.085, 14.4, 0.015, 49.98, 0.62, 95.3)


predicted_fault, predicted_cause = predict_fault_and_cause(input_data)

print(f"Predicted Fault: {predicted_fault}")
print(f"Cause: {predicted_cause}")


# In[68]:


#Use histograms or boxplots to visualize distributions and detect outliers.

import seaborn as sns
import matplotlib.pyplot as plt

for col in X_train.columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(X_train[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()


# In[69]:


#Identify multicollinearity or highly correlated features.

plt.figure(figsize=(10, 8))
sns.heatmap(X_train.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()


# In[70]:


#Analyze how individual features impact the target.

for col in X_train.columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=y_train, y=X_train[col])
    plt.title(f'{col} vs Target')
    plt.show()


# In[71]:


#Evaluate model performance on classification tasks.

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()


# In[103]:


import joblib
rf_classifier.fault_cause_mapping = fault_cause_mapping
joblib.dump(rf_classifier, "rf_model.pkl")


# In[ ]:





# In[ ]:


# dummy


# In[56]:


print("Feature names used during training:", rf_classifier.feature_names_in_)


# In[2]:


import sklearn
print(sklearn.__version__)


# In[1]:


import numpy
import sklearn
print("Numpy version:", numpy.__version__)
print("Scikit-learn version:", sklearn.__version__)


# In[5]:


import numpy as np
import pandas as pd
import sklearn
import joblib
import requests
import time
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Print versions of the libraries
print("Numpy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("Scikit-learn version:", sklearn.__version__)  # Corrected line
print("Joblib version:", joblib.__version__)
print("Requests version:", requests.__version__)
print("Streamlit version:", st.__version__)

