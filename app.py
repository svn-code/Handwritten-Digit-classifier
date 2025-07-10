import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load models using pickle
with open("best_svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("best_knn_model.pkl", "rb") as f:
    knn_model = pickle.load(f)

with open("best_decision_tree_model.pkl", "rb") as f:
    tree_model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Model dictionary
models = {
    "SVM": svm_model,
    "KNN": knn_model,
    "Decision Tree": tree_model
}

# App layout
st.set_page_config(page_title="Digit Classifier", layout="centered")
st.title("✍️ Draw a Digit to Predict")
st.markdown("Draw any digit (0–9) on the canvas below, and click **Predict** to see what the model thinks!")

# Sidebar: model selection
st.sidebar.title("⚙️ Settings")
model_choice = st.sidebar.selectbox("Choose Model", list(models.keys()))

# Canvas for drawing digits
canvas_result = st_canvas(
    fill_color="#000000",  # Black background
    stroke_width=10,
    stroke_color="#FFFFFF",  # White drawing
    background_color="#000000",
    height=192,
    width=192,
    drawing_mode="freedraw",
    key="canvas",
)

# Predict Button
if st.button("🎯 Predict Digit"):
    if canvas_result.image_data is not None:
        # Convert canvas image to grayscale
        image = Image.fromarray((canvas_result.image_data[:, :, 0:3]).astype(np.uint8))
        image = image.convert("L")  # Convert to grayscale
        image = image.resize((8, 8), resample=Image.Resampling.LANCZOS)

        # Display what the model sees
        st.subheader("🖼️ What the model sees (8×8 input)")
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        st.pyplot(fig)

        # Prepare input for prediction
        img_array = np.array(image).astype(np.float64)
        flat = img_array.flatten()
        scaled_pixel = (flat / 255.0) * 16.0
        X_input = scaled_pixel.reshape(1, -1)
        X_scaled = scaler.transform(X_input)

        # Predict
        model = models[model_choice]
        prediction = model.predict(X_scaled)[0]

        # Show prediction
        st.markdown("---")
        st.markdown(
            f"<h2 style='text-align: center;'>🎯 Predicted Digit: <span style='color: green;'>{prediction}</span></h2>",
            unsafe_allow_html=True
        )
        st.success(f"Prediction made using **{model_choice}** model")
    else:
        st.warning("⚠️ Please draw a digit on the canvas first.")
