import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# Page Config
st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")

st.title("Breast Cancer Prediction System")
st.markdown("Real-World Use Cases of ML")


st.sidebar.header("Patient Data Input")

def user_input_features():

    radius_mean = st.sidebar.slider('Radius Mean', 6.0, 30.0, 15.0)
    texture_mean = st.sidebar.slider('Texture Mean', 9.0, 40.0, 20.0)
    perimeter_mean = st.sidebar.slider('Perimeter Mean', 40.0, 190.0, 90.0)
    area_mean = st.sidebar.slider('Area Mean', 140.0, 2500.0, 600.0)
    smoothness_mean = st.sidebar.slider('Smoothness Mean', 0.05, 0.25, 0.1)
    
    data = {
        'mean radius': radius_mean,
        'mean texture': texture_mean,
        'mean perimeter': perimeter_mean,
        'mean area': area_mean,
        'mean smoothness': smoothness_mean,
        # Fillers for the rest 
        'mean compactness': 0.1, 'mean concavity': 0.09, 'mean concave points': 0.05,
        'mean symmetry': 0.18, 'mean fractal dimension': 0.06,
        'radius error': 0.4, 'texture error': 1.2, 'perimeter error': 2.8,
        'area error': 40.0, 'smoothness error': 0.007, 'compactness error': 0.025,
        'concavity error': 0.03, 'concave points error': 0.01, 'symmetry error': 0.02,
        'fractal dimension error': 0.004, 'worst radius': radius_mean * 1.2,
        'worst texture': texture_mean * 1.2, 'worst perimeter': perimeter_mean * 1.2,
        'worst area': area_mean * 1.2, 'worst smoothness': 0.13,
        'worst compactness': 0.25, 'worst concavity': 0.27, 'worst concave points': 0.11,
        'worst symmetry': 0.29, 'worst fractal dimension': 0.08
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display User Input
st.subheader('1. Patient Parameters')
st.write(input_df[['mean radius', 'mean texture', 'mean perimeter', 'mean area']])

# Load Model & Scaler
try:
    model = pickle.load(open('best_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    
    # Scale Input 
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    st.subheader('2. Prediction Result')
    if prediction[0] == 1:
        st.error(f"**MALIGNANT** (Cancerous)")
        st.write(f"Confidence: {prediction_proba[0][1]*100:.2f}%")
    else:
        st.success(f"**BENIGN** (Safe)")
        st.write(f"Confidence: {prediction_proba[0][0]*100:.2f}%")

except FileNotFoundError:
    st.warning("Please run 'train_model.py' first to generate the model files!")

# --- Analysis Tabs ---
# st.markdown("---")
# st.subheader("3. Technical Analysis")
# tab1, tab2 , tab3 = st.tabs(["PCA Visualization", "Neural Network Loss", "Confusion Matrix"])

# with tab1:
#     st.write("Dimensionality Reduction: Compressing 30 features into 2D space.")
#     try:
#         image = Image.open('pca_plot.png')
#         st.image(image, caption='PCA Projection of Cancer Dataset')
#     except:
#         st.write("Run training script to generate plot.")

# with tab2:
#     st.write("Deep Learning Training Curve: Checking for Overfitting/Underfitting.")
#     try:
#         image = Image.open('nn_loss.png')
#         st.image(image, caption='Training vs Validation Loss')
#     except:
#         st.write("Run training script to generate plot.")

# --- Analysis Tabs ---
st.markdown("---")
st.subheader("3. Technical Analysis (Syllabus Topics)")
tab1, tab2, tab3 = st.tabs([
    "PCA Visualization",
    "Neural Network Loss",
    "Confusion Matrix"
])

# PCA TAB
with tab1:
    st.write("Dimensionality Reduction: Compressing 30 features into 2D space.")
    try:
        image = Image.open('pca_plot.png')
        st.image(image, caption='PCA Projection of Cancer Dataset')
    except:
        st.write("Run training script to generate PCA plot.")

# Neural Network Loss TAB
with tab2:
    st.write("Deep Learning Training Curve: Checking for Overfitting/Underfitting.")
    try:
        image = Image.open('nn_loss.png')
        st.image(image, caption='Training vs Validation Loss')
    except:
        st.write("Run training script to generate loss plot.")

# Confusion Matrix TAB
with tab3:
    st.write("Confusion Matrix: Understanding True/False Positives & Negatives")
    try:
        image = Image.open('confusion_matrix.png')
        st.image(image, caption='Confusion Matrix', use_container_width=True)
    except:
        st.warning("Confusion matrix not found. Run train_model.py to generate confusion_matrix.png")
