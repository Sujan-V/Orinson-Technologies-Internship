import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load Iris dataset and train the model
@st.cache_resource
def train_model():
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data['target'] = iris.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data.iloc[:, :-1])
    y = data['target']

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_scaled, y)

    # Save the scaler and model
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(model, "model.pkl")

    return iris

# Load model and scaler
iris = train_model()
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

# Streamlit App
st.title("Iris Flower Classification 🌸")
st.write("Predict the type of Iris flower based on its features.")

# User input for features
sepal_length = st.slider("Sepal Length (cm)", float(iris.data[:, 0].min()), float(iris.data[:, 0].max()))
sepal_width = st.slider("Sepal Width (cm)", float(iris.data[:, 1].min()), float(iris.data[:, 1].max()))
petal_length = st.slider("Petal Length (cm)", float(iris.data[:, 2].min()), float(iris.data[:, 2].max()))
petal_width = st.slider("Petal Width (cm)", float(iris.data[:, 3].min()), float(iris.data[:, 3].max()))

# Prediction
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    predicted_class = iris.target_names[prediction[0]]
    st.write(f"The predicted class is: **{predicted_class}**")
