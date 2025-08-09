# Multiple-Disease-prediction


🩺 Multi-Disease Prediction System
📌 Project Overview
The Multi-Disease Prediction System is a web application built using Python, Machine Learning, and Streamlit that predicts the likelihood of three different diseases:

Chronic Kidney Disease (CKD)

Indian Liver Patient Disease (ILPD)

Parkinson’s Disease

The app uses pre-trained machine learning models for each disease and allows users to select the disease they want to check from a sidebar Option Menu.

🚀 Features
✅ Predicts Kidney Disease based on medical test results like Age, Blood Pressure, Sugar Levels, Creatinine, Hemoglobin, etc.
✅ Predicts Indian Liver Disease using Age, Gender, Albumin, Bilirubin, and enzyme levels.
✅ Predicts Parkinson’s Disease using voice measurement parameters.
✅ User-friendly Streamlit dashboard with a clean design.
✅ Single application for all three diseases using Option Menu navigation.
✅ Instant predictions with pre-trained models.

🛠 Tech Stack
Python 3

Pandas & NumPy (Data Handling)

Scikit-learn (Model Training)

Streamlit (Web App)

Pickle (Model Saving & Loading)

Option Menu for Navigation

Project Structure
Multi-Disease-Prediction/
│
├── kidney_model.pkl                # Trained model for CKD
├── liver_model.pkl                  # Trained model for ILPD
├── parkinson_model.pkl              # Trained model for Parkinson's
├── app.py                           # Main Streamlit app
├── kidney_disease.csv               # Dataset for CKD
├── indian_liver_patient.csv         # Dataset for ILPD
├── parkinsons.csv                   # Dataset for Parkinson’s               
└── README.md                        # Project documentation
