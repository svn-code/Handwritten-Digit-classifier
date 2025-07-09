import streamlit as st
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Load models with pickle
with open("best_svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("best_knn_model.pkl", "rb") as f:
    knn_model = pickle.load(f)

with open("best_decision_tree_model.pkl", "rb") as f:
    tree_model = pickle.load(f)

models = {
    "SVM": svm_model,
    "KNN": knn_model,
    "Decision Tree": tree_model
}

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Sidebar inputs
st.sidebar.title("⚙️ Settings")
model_choice = st.sidebar.selectbox("Choose Model", list(models.keys()))
uploaded_file = st.sidebar.file_uploader("📤 Upload a digit image", type=["jpg", "jpeg", "png"])

# Main title
st.title("🔢 Digit Classifier")
st.markdown(
    "Upload a **handwritten digit image** (any size, grayscale or color) and let the classifier predict the number (0-9)."
)

# If file uploaded
if uploaded_file:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    img_array = np.array(image)

    # Resize to 8x8
    resized = cv2.resize(img_array, (8, 8), interpolation=cv2.INTER_AREA)

    # Flatten & normalize to match digit dataset (0–16 scale)
    flat = resized.flatten().astype(np.float64)
    scaled_pixel = (flat / 255.0) * 16.0
    X_input = scaled_pixel.reshape(1, -1)

    # Scale using original scaler
    X_scaled = scaler.transform(X_input)

    # Predict
    model = models[model_choice]
    prediction = model.predict(X_scaled)[0]

    # Show results
    st.subheader("🖼️ What the model sees (8×8 input image)")
    fig, ax = plt.subplots()
    ax.imshow(resized, cmap='gray')
    ax.axis('off')
    st.pyplot(fig)

    # Show prediction
    st.markdown("---")
    st.markdown(f"<h2 style='text-align: center;'>🎯 Predicted Digit: <span style='color: green;'>{prediction}</span></h2>",
                unsafe_allow_html=True)
    st.success(f"Prediction made using **{model_choice}** model")

else:
    st.info("👈 Upload a digit image in the sidebar to begin.")
