#_________________________________________________________________________________________________________________________________________________
#Import Libraries---------------------------------------------------------------------------------------------------------------------------------
import pandas as  pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report,mean_squared_error
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

#_________________________________________________________________________________________________________________________________________________
# Load Dataset and check null values -------------------------------------------------------------------------------------------------------------

parkinsons = pd.read_csv("C:/Users/Yokesh/guvi mini projects/Multiple-Disease-prediction/parkinsons - parkinsons.csv")

print(parkinsons)

parkinsons.info()

#_________________________________________________________________________________________________________________________________________________
# EDA Bar Plot — Count of Target Variable (status)------------------------------------------------------------------------------------------------
parkinsons['status'].value_counts().plot(kind='bar')
plt.title("Distribution of Status (0 = Healthy, 1 = Parkinson's)")
plt.xlabel("Status")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.grid(True)
plt.savefig("count_target_variable.png")
plt.show()

#Scatter Plot — MDVP:Fo(Hz) vs MDVP:Shimmer colored by status-------------------------------------------------------------------------------------

plt.figure(figsize=(8, 5))
for status in parkinsons['status'].unique():
    subset = parkinsons[parkinsons['status'] == status]
    plt.scatter(subset['MDVP:Fo(Hz)'], subset['MDVP:Shimmer'], label=f'Status {status}', alpha=0.6)

plt.xlabel('MDVP:Fo(Hz) - Fundamental Frequency')
plt.ylabel('MDVP:Shimmer')
plt.title("Fo(Hz) vs Shimmer (by Status)")
plt.legend(title='Status')
plt.grid(True)
plt.savefig("parkinsons_scatterplot_mdvp.png")
plt.show()
#_________________________________________________________________________________________________________________________________________________
# Drop 'name' column (not a feature)--------------------------------------------------------------------------------------------------------------
df = parkinsons.drop(['name'], axis=1)

#_________________________________________________________________________________________________________________________________________________
# Features & target-------------------------------------------------------------------------------------------------------------------------------
X = df.drop('status', axis=1)
y = df['status']

# Train/test split--------------------------------------------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models-----------------------------------------------------------------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Evaluate models---------------------------------------------------------------------------------------------------------------------------------
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    results[name] = {
        "model": model,
        "accuracy": acc,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "mse": mse
    }

    print(f"🔹 {name}")
    print(f"   - Accuracy : {acc:.4f}")
    print(f"   - F1 Score : {f1:.4f}")
    print(f"   - Precision: {precision:.4f}")
    print(f"   - Recall   : {recall:.4f}")
    print(f"   - MSE      : {mse:.4f}\n")

#_________________________________________________________________________________________________________________________________________________
# Select best model based on F1-score-------------------------------------------------------------------------------------------------------------
best_model_name = max(results, key=lambda x: results[x]['f1_score'])
best_model = results[best_model_name]['model']

print(f" Best Model: {best_model_name} with F1 Score = {results[best_model_name]['f1_score']:.4f}")
#_________________________________________________________________________________________________________________________________________________
# Save the best model-----------------------------------------------------------------------------------------------------------------------------
with open("parkinsons_model.pkl", "wb") as f:
    pickle.dump(best_model, f)


#**************************************************************END********************************************************************************



