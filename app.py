import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import os

# Path to save/load the model
MODEL_PATH = "line_orientation_model.joblib"

# Load model if it exists
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    st.session_state.model = model

# Initialize session state for data if not already done
if "data" not in st.session_state:
    st.session_state.data = []
    st.session_state.labels = []

st.title("Draw a Line: Train or Predict Orientation")
st.write("Draw a line on the canvas below, then choose to **Train** the model or **Predict** the line's orientation.")

# Canvas for drawing
canvas_result = st_canvas(
    fill_color="white",  # Background color
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=100,
    width=100,
    drawing_mode="freedraw",
    key="canvas",
)

# Capture drawing and convert to image array
if canvas_result.image_data is not None:
    img_array = canvas_result.image_data[:, :, 0]  # Get grayscale values only

# Options to train or predict
action = st.selectbox("Choose action:", ["Predict", "Train"])

if action == "Train":
    # Choose label for training
    label = st.radio("Label this line as:", ["Horizontal", "Vertical"])

    # Save the drawing and label
    if st.button("Save Drawing"):
        if canvas_result.image_data is not None:
            st.session_state.data.append(img_array.flatten())  # Flatten the image
            st.session_state.labels.append(0 if label == "Horizontal" else 1)  # Label as 0 or 1
            st.success("Drawing saved for training!")

    # Train the model
    if st.button("Train Model"):
        if len(st.session_state.data) < 2:
            st.warning("Need at least two drawings to train the model.")
        else:
            # Prepare data and labels
            X = np.array(st.session_state.data)
            y = np.array(st.session_state.labels)

            # Split data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Train a logistic regression model
            model = LogisticRegression()
            model.fit(X_train, y_train)

            # Save the model to disk
            joblib.dump(model, MODEL_PATH)
            st.session_state.model = model

            # Display model accuracy
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Model accuracy: {accuracy * 100:.2f}%")
            st.success("Model trained and saved successfully!")

elif action == "Predict":
    # Predict with the model if available
    if "model" in st.session_state:
        if st.button("Predict Line Orientation"):
            X_new = img_array.flatten().reshape(1, -1)  # Flatten and reshape for prediction
            prediction = st.session_state.model.predict(X_new)
            prediction_label = "Horizontal" if prediction[0] == 0 else "Vertical"
            st.write(f"Predicted line orientation: **{prediction_label}**")
    else:
        st.warning("Model not trained yet. Please add data and train first.")
