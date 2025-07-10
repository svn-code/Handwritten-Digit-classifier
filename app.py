import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load models
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

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Sidebar
st.sidebar.title("⚙️ Settings")
model_choice = st.sidebar.selectbox("Choose Model", list(models.keys()))

# App title
st.title("✍️ Draw a Digit to Classify (0–9)")
st.markdown("Use the canvas below to draw a digit with your mouse or touch!")

# Canvas settings
canvas_result = st_canvas(
    fill_color="#000000",  # Drawing color
    stroke_width=10,
    stroke_color="#FFFFFF",  # White strokes
    background_color="#000000",  # Black canvas
    height=192,
    width=192,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:
    # Convert canvas to image
    image = Image.fromarray((canvas_result.image_data[:, :, 0:3]).astype(np.uint8))
    image = image.convert("L")  # Grayscale
    image = image.resize((8, 8), resample=Image.Resampling.LANCZOS)

    # Display what the model sees
    st.subheader("🖼️ What the model sees (8×8 image)")
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    st.pyplot(fig)

    # Flatten, scale 0–16, predict
    img_array = np.array(image).astype(np.float64)
    flat = img_array.flatten()
    scaled_pixel = (flat / 255.0) * 16.0
    X_input = scaled_pixel.reshape(1, -1)
    X_scaled = scaler.transform(X_input)

    # Predict
    model = models[model_choice]
    prediction = model.predict(X_scaled)[0]

    # Show result
    st.markdown("---")
    st.markdown(f"<h2 style='text-align: center;'>🎯 Predicted Digit: <span style='color: green;'>{prediction}</span></h2>",
                unsafe_allow_html=True)
    st.success(f"Prediction made using **{model_choice}** model")

else:
    st.info("Draw a digit on the canvas to get a prediction.")
