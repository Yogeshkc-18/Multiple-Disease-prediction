#_________________________________________________________________________________________________________________________________________________
#  Import Libraries-------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

#_________________________________________________________________________________________________________________________________________________
#Load the dataset---------------------------------------------------------------------------------------------------------------------------------

ind_liver = pd.read_csv("C:/Users/Yokesh/guvi mini projects/Multiple-Disease-prediction/indian_liver_patient - indian_liver_patient.csv")

#_________________________________________________________________________________________________________________________________________________
# check the dataset (null values / duplicates / datatypes)----------------------------------------------------------------------------------------

ind_liver.info()
ind_liver.isna()
ind_liver.drop_duplicates()

print(ind_liver)


ind_liver['Albumin_and_Globulin_Ratio'].fillna(ind_liver['Albumin_and_Globulin_Ratio'].mean(), inplace=True)

ind_liver.info()
#_________________________________________________________________________________________________________________________________________________
# EDA -----------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(7,5))
sns.countplot(x='Gender', hue='Dataset', data=ind_liver)
plt.title("Liver Disease by Gender")
plt.savefig("eda1_gender_liver.png")
plt.show()

plt.figure(figsize=(7,5))
sns.histplot(data=ind_liver, x='Age', hue='Dataset', kde=True, bins=30)
plt.title("Age Distribution by Liver Disease")
plt.savefig("eda2_age_distribution.png")
plt.show()
#_________________________________________________________________________________________________________________________________________________
# Encode Categorical Column-----------------------------------------------------------------------------------------------------------------------
le = LabelEncoder()
ind_liver['Gender'] = le.fit_transform(ind_liver['Gender'])
print(ind_liver)
#_________________________________________________________________________________________________________________________________________________
# Convert target to binary (1 = liver disease, 0 = no disease)------------------------------------------------------------------------------------

ind_liver['Dataset'] = ind_liver['Dataset'].map({1: 1, 2: 0})

#_________________________________________________________________________________________________________________________________________________
# Select Important Features-----------------------------------------------------------------------------------------------------------------------
important_features = ['Age', 'Total_Bilirubin', 'Direct_Bilirubin',
                      'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
                      'Aspartate_Aminotransferase', 'Total_Protiens',
                      'Albumin', 'Albumin_and_Globulin_Ratio', 'Gender']

X = ind_liver[important_features]
y = ind_liver['Dataset']

print(X,y)
#_________________________________________________________________________________________________________________________________________________
# Train-Test Split--------------------------------------------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#_________________________________________________________________________________________________________________________________________________
# Balance the dataset-----------------------------------------------------------------------------------------------------------------------------
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

#_________________________________________________________________________________________________________________________________________________
# Split data--------------------------------------------------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)


#_________________________________________________________________________________________________________________________________________________
# Model selection with hyperparameter tuning (XGBoost)--------------------------------------------------------------------------------------------
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
params = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 150]
}

grid = GridSearchCV(xgb, params, cv=3, scoring='accuracy')
grid.fit(X_train, y_train)
#_________________________________________________________________________________________________________________________________________________
# Best model--------------------------------------------------------------------------------------------------------------------------------------
best_model = grid.best_estimator_

# Predictions-------------------------------------------------------------------------------------------------------------------------------------
y_pred = best_model.predict(X_test)

# Metrics-----------------------------------------------------------------------------------------------------------------------------------------
acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Accuracy:", acc)
print("Precision:", pre)
print("Recall:", rec)
print("F1 Score:", f1)
print("MSE:", mse)

#_________________________________________________________________________________________________________________________________________________
# Save model
with open('indian_liver_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++END++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
