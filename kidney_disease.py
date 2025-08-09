#________________________________________________________________________________________________________________________________________________
#  Import Libraries------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

kid_disease = pd.read_csv("C:/Users/Yokesh/guvi mini projects/Multiple-Disease-prediction/kidney_disease - kidney_disease.csv")

print(kid_disease)

kid_disease.info()
#_________________________________________________________________________________________________________________________________________________
#  Clean Data-------------------------------------------------------------------------------------------------------------------------------------
kid_disease.dropna(how='all', inplace=True)
kid_disease.replace('?', np.nan, inplace=True)

kid_disease.info()

#_________________________________________________________________________________________________________________________________________________
# -------------------- Chart 1: Bar plot ---------------------------------------------------------------------------------------------------------
# Count of CKD vs Not CKD-------------------------------------------------------------------------------------------------------------------------
ckd_count = kid_disease['classification'].value_counts()
plt.figure(figsize=(5, 4))
ckd_count.plot(kind='bar', color=['green', 'red'])
plt.title('CKD vs Not CKD')
plt.xlabel('Classification')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("count_of_ckd_vs_not_ckd.png")
plt.show()

# -------------------- Chart 2: Scatter plot -----------------------------------------------------------------------------------------------------
# Scatter plot: Serum Creatinine vs Hemoglobin----------------------------------------------------------------------------------------------------
plt.figure(figsize=(5, 4))
plt.scatter(kid_disease['sc'], kid_disease['hemo'], alpha=0.7)
plt.title("Serum Creatinine vs Hemoglobin")
plt.xlabel("Serum Creatinine")
plt.ylabel("Hemoglobin")
plt.tight_layout()
plt.savefig("serum_creatinine_vs_Hemoglobin.png")
plt.show()

#_________________________________________________________________________________________________________________________________________________
# Convert specific columns to numeric-------------------------------------------------------------------------------------------------------------
numeric_cols = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
for col in numeric_cols:
    kid_disease[col] = pd.to_numeric(kid_disease[col], errors='coerce')

# Fill missing values-----------------------------------------------------------------------------------------------------------------------------
for col in kid_disease.columns:
    if kid_disease[col].dtype == 'object':
        kid_disease[col].fillna(kid_disease[col].mode()[0], inplace=True)
    else:
        kid_disease[col].fillna(kid_disease[col].mean(), inplace=True)


kid_disease.info()

print(kid_disease)

#_________________________________________________________________________________________________________________________________________________
# Label encode categorical columns----------------------------------------------------------------------------------------------------------------
le = LabelEncoder()
for col in kid_disease.select_dtypes(include='object').columns:
    kid_disease[col] = le.fit_transform(kid_disease[col])

# Map target variable: ckd = 1, notckd = 0--------------------------------------------------------------------------------------------------------
kid_disease['classification'] = kid_disease['classification'].apply(lambda x: 1 if x == 1 or x == 'ckd' else 0)

# Feature selection - manually selecting important ones-------------------------------------------------------------------------------------------
selected_features = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
X = kid_disease[selected_features]
y = kid_disease['classification']

print(kid_disease)
print(X)
print(y)

#_________________________________________________________________________________________________________________________________________________
# Split data--------------------------------------------------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models------------------------------------------------------------------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results ={}
#_________________________________________________________________________________________________________________________________________________
# Evaluate models---------------------------------------------------------------------------------------------------------------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    
    print(f"\n{name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
 
    results[name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
       'F1-Score': f1
       }
    
#_________________________________________________________________________________________________________________________________________________    
# Save the Best Model-----------------------------------------------------------------------------------------------------------------------------
best_model_name = max(results, key=lambda x: results[x]['F1-Score'])
best_model = models[best_model_name]

with open("best_kidney_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print(f"\n Best model ({best_model_name}) saved as 'best_kidney_model.pkl'")

#====================================================================END==========================================================================