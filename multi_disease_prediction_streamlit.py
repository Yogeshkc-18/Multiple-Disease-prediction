import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np

# --------------------------
# LOAD TRAINED MODELS
# --------------------------
kidney_model = pickle.load(open("best_kidney_model.pkl", "rb"))
liver_model = pickle.load(open("indian_liver_model.pkl", "rb"))
parkinson_model = pickle.load(open("parkinsons_model.pkl", "rb"))

st.set_page_config(
    page_title="Multiple Disease Prediction",
    page_icon="🩺",  # You can use an emoji or path to an image file
    layout="wide"
)
# Sidebar Menu
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Kidney Disease Prediction',
         'Liver Disease Prediction',
         'Parkinson’s Disease Prediction'],
        menu_icon='hospital-fill',
        icons=['droplet', 'capsule', 'activity'],
        default_index=0
    )

# Kidney Disease Prediction Page
if selected == 'Kidney Disease Prediction':
    st.title('Kidney Disease Prediction')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=1, max_value=100, value=45)
    with col2:
        bp = st.number_input('Blood Pressure', min_value=50, max_value=200, value=80)
    with col3:
        bgr = st.number_input('Blood Glucose Random', min_value=50, max_value=500, value=120)
    with col1:
        bu = st.number_input('Blood Urea', min_value=1.0, max_value=200.0, value=40.0)
    with col2:
        sc = st.number_input('Serum Creatinine', min_value=0.1, max_value=15.0, value=1.2)
    with col3:
        sod = st.number_input('Sodium', min_value=100.0, max_value=160.0, value=140.0)
    with col1:
        pot = st.number_input('Potassium', min_value=2.0, max_value=8.0, value=4.5)
    with col2:
        hemo = st.number_input('Hemoglobin', min_value=3.0, max_value=20.0, value=13.0)
    with col3:
        pcv = st.number_input('Packed Cell Volume', min_value=20, max_value=60, value=40)
    with col1:
        wc = st.number_input('White Blood Cell Count', min_value=2000, max_value=20000, value=8000)
    with col2:
        rc = st.number_input('Red Blood Cell Count', min_value=2.0, max_value=8.0, value=5.0)

    kidney_result = ''
    if st.button('Predict Kidney Disease'):
        features = np.array([[age, bp, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc]])
        pred = kidney_model.predict(features)
        kidney_result = 'Kidney Disease Detected' if pred[0] == 1 else 'No Kidney Disease'

    st.success(kidney_result)






    # Liver Disease Prediction Page
if selected == 'Liver Disease Prediction':
    st.title('Indian Liver Disease Prediction')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=1, max_value=100, value=45)
    with col2:
        gender = st.selectbox('Gender', ['Male', 'Female'])
    with col3:
        total_bilirubin = st.number_input('Total Bilirubin', min_value=0.0, max_value=50.0, value=1.0)

    with col1:
        direct_bilirubin = st.number_input('Direct Bilirubin', min_value=0.0, max_value=30.0, value=0.5)
    with col2:
        alkaline_phosphotase = st.number_input('Alkaline Phosphotase', min_value=0, max_value=2000, value=200)
    with col3:
        alamine_aminotransferase = st.number_input('Alamine Aminotransferase', min_value=0, max_value=1000, value=30)

    with col1:
        aspartate_aminotransferase = st.number_input('Aspartate Aminotransferase', min_value=0, max_value=1000, value=40)
    with col2:
        total_proteins = st.number_input('Total Proteins', min_value=0.0, max_value=10.0, value=6.5)
    with col3:
        albumin = st.number_input('Albumin', min_value=0.0, max_value=6.0, value=3.0)

    with col1:
        albumin_globulin_ratio = st.number_input('Albumin and Globulin Ratio', min_value=0.0, max_value=3.0, value=1.0)

    # Gender Encoding
    gender_value = 1 if gender == 'Male' else 0

    liver_result = ''
    if st.button('Predict Liver Disease'):
        features = np.array([[age, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
                              alamine_aminotransferase, aspartate_aminotransferase,
                              total_proteins, albumin, albumin_globulin_ratio, gender_value]])
        pred = liver_model.predict(features)
        liver_result = '⚠️ Liver Disease Detected' if pred[0] == 1 else '✅ No Liver Disease'

    st.success(liver_result)




# Parkinson's Disease Prediction Page
if selected == "Parkinson’s Disease Prediction":
    st.title("Parkinson’s Disease Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        fo = st.number_input('MDVP:Fo(Hz)', min_value=0.0, max_value=300.0, value=120.0)
    with col2:
        fhi = st.number_input('MDVP:Fhi(Hz)', min_value=0.0, max_value=600.0, value=150.0)
    with col3:
        flo = st.number_input('MDVP:Flo(Hz)', min_value=0.0, max_value=300.0, value=100.0)

    with col1:
        jitter_percent = st.number_input('MDVP:Jitter(%)', min_value=0.0, max_value=1.0, value=0.005)
    with col2:
        jitter_abs = st.number_input('MDVP:Jitter(Abs)', min_value=0.0, max_value=1.0, value=0.00005)
    with col3:
        rap = st.number_input('MDVP:RAP', min_value=0.0, max_value=1.0, value=0.003)

    with col1:
        ppq = st.number_input('MDVP:PPQ', min_value=0.0, max_value=1.0, value=0.0035)
    with col2:
        ddp = st.number_input('Jitter:DDP', min_value=0.0, max_value=1.0, value=0.009)
    with col3:
        shimmer = st.number_input('MDVP:Shimmer', min_value=0.0, max_value=1.0, value=0.03)

    with col1:
        shimmer_db = st.number_input('MDVP:Shimmer(dB)', min_value=0.0, max_value=10.0, value=0.3)
    with col2:
        apq3 = st.number_input('Shimmer:APQ3', min_value=0.0, max_value=1.0, value=0.02)
    with col3:
        apq5 = st.number_input('Shimmer:APQ5', min_value=0.0, max_value=1.0, value=0.02)

    with col1:
        apq = st.number_input('MDVP:APQ', min_value=0.0, max_value=1.0, value=0.03)
    with col2:
        dda = st.number_input('Shimmer:DDA', min_value=0.0, max_value=1.0, value=0.06)
    with col3:
        nhr = st.number_input('NHR', min_value=0.0, max_value=1.0, value=0.02)

    with col1:
        hnr = st.number_input('HNR', min_value=0.0, max_value=50.0, value=20.0)
    with col2:
        rpde = st.number_input('RPDE', min_value=0.0, max_value=1.0, value=0.5)
    with col3:
        dfa = st.number_input('DFA', min_value=0.0, max_value=1.0, value=0.6)

    with col1:
        spread1 = st.number_input('Spread1', min_value=-10.0, max_value=10.0, value=-4.0)
    with col2:
        spread2 = st.number_input('Spread2', min_value=0.0, max_value=5.0, value=0.3)
    with col3:
        d2 = st.number_input('D2', min_value=0.0, max_value=5.0, value=2.3)

    with col1:
        ppe = st.number_input('PPE', min_value=0.0, max_value=1.0, value=0.2)

    parkinson_result = ''
    if st.button('Predict Parkinson’s Disease'):
        features = np.array([[fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
                               shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
                               rpde, dfa, spread1, spread2, d2, ppe]])
        pred = parkinson_model.predict(features)
        parkinson_result = '⚠️ Parkinson’s Disease Detected' if pred[0] == 1 else '✅ No Parkinson’s Disease'

    st.success(parkinson_result)
