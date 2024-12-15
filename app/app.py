import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt  # Import matplotlib for pie chart
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Load your data, models, and scaler here
logistic_model = pickle.load(open("../model/model.pkl", "rb"))
random_forest_model = pickle.load(open("../model/random_forest_model.pkl", "rb"))
scaler = pickle.load(open("../model/scaler.pkl", "rb"))

# Sidebar for model selection
st.sidebar.title("Model Selection")
model_choice = st.sidebar.selectbox("Choose the Machine Learning Model", ("Logistic Regression", "Random Forest"))

# Sidebar inputs for all features expected by the model
st.sidebar.title("Cell Nuclei Measurements (Mean, SE, Worst)")
radius_mean = st.sidebar.slider("Radius (mean)", 0.0, 28.11, 14.13)
texture_mean = st.sidebar.slider("Texture (mean)", 0.0, 39.28, 29.23)
perimeter_mean = st.sidebar.slider("Perimeter (mean)", 0.0, 188.5, 91.97)
area_mean = st.sidebar.slider("Area (mean)", 0.0, 2501.0, 654.89)
smoothness_mean = st.sidebar.slider("Smoothness (mean)", 0.0, 0.16, 0.1)
compactness_mean = st.sidebar.slider("Compactness (mean)", 0.0, 0.35, 0.1)
concavity_mean = st.sidebar.slider("Concavity (mean)", 0.0, 0.45, 0.09)
concave_points_mean = st.sidebar.slider("Concave Points (mean)", 0.0, 0.3, 0.1)
symmetry_mean = st.sidebar.slider("Symmetry (mean)", 0.0, 0.3, 0.1)
fractal_dimension_mean = st.sidebar.slider("Fractal Dimension (mean)", 0.0, 0.1, 0.06)

radius_se = st.sidebar.slider("Radius (SE)", 0.0, 3.0, 0.2)
texture_se = st.sidebar.slider("Texture (SE)", 0.0, 5.0, 1.0)
perimeter_se = st.sidebar.slider("Perimeter (SE)", 0.0, 21.0, 2.0)
area_se = st.sidebar.slider("Area (SE)", 0.0, 542.0, 40.0)
smoothness_se = st.sidebar.slider("Smoothness (SE)", 0.0, 0.03, 0.02)
compactness_se = st.sidebar.slider("Compactness (SE)", 0.0, 0.135, 0.03)
concavity_se = st.sidebar.slider("Concavity (SE)", 0.0, 0.4, 0.05)
concave_points_se = st.sidebar.slider("Concave Points (SE)", 0.0, 0.05, 0.02)
symmetry_se = st.sidebar.slider("Symmetry (SE)", 0.0, 0.08, 0.02)
fractal_dimension_se = st.sidebar.slider("Fractal Dimension (SE)", 0.0, 0.03, 0.01)

radius_worst = st.sidebar.slider("Radius (worst)", 0.0, 50.0, 16.0)
texture_worst = st.sidebar.slider("Texture (worst)", 0.0, 50.0, 25.0)
perimeter_worst = st.sidebar.slider("Perimeter (worst)", 0.0, 250.0, 100.0)
area_worst = st.sidebar.slider("Area (worst)", 0.0, 4254.0, 880.0)
smoothness_worst = st.sidebar.slider("Smoothness (worst)", 0.0, 0.22, 0.1)
compactness_worst = st.sidebar.slider("Compactness (worst)", 0.0, 1.58, 0.3)
concavity_worst = st.sidebar.slider("Concavity (worst)", 0.0, 1.25, 0.3)
concave_points_worst = st.sidebar.slider("Concave Points (worst)", 0.0, 0.3, 0.1)
symmetry_worst = st.sidebar.slider("Symmetry (worst)", 0.0, 0.6, 0.3)
fractal_dimension_worst = st.sidebar.slider("Fractal Dimension (worst)", 0.0, 0.2, 0.1)

# Prepare the input data with correct column names
input_data = pd.DataFrame([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
                            compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
                            radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se,
                            concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
                            radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst,
                            compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst]], 
                          columns=["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
                                   "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
                                   "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se",
                                   "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
                                   "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
                                   "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"])

# Scale the input data
input_data = scaler.transform(input_data)

# Display prediction based on the selected model
st.title("Breast Cancer Predictor")

if model_choice == "Logistic Regression":
    try:
        prediction = logistic_model.predict(input_data)
        probability = logistic_model.predict_proba(input_data)
        model_name = "Logistic Regression"
    except Exception as e:
        st.error(f"Error with Logistic Regression: {e}")
        st.stop()

elif model_choice == "Random Forest":
    try:
        prediction = random_forest_model.predict(input_data)
        probability = random_forest_model.predict_proba(input_data)
        model_name = "Random Forest"
    except Exception as e:
        st.error(f"Error with Random Forest: {e}")
        st.stop()

# Display results
st.subheader(f"Cell Cluster Prediction using {model_name}")
if prediction[0] == 1:
    st.write("The cell cluster is **Malignant**")
else:
    st.write("The cell cluster is **Benign**")

st.write(f"Probability of being benign: {probability[0][0]:.2f}")
st.write(f"Probability of being malignant: {probability[0][1]:.2f}")

# Display a pie chart of probabilities
st.subheader("Prediction Probability Chart")
fig, ax = plt.subplots()
ax.pie(probability[0], labels=["Benign", "Malignant"], autopct='%1.1f%%', startangle=90, colors=["skyblue", "salmon"])
ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(fig)
