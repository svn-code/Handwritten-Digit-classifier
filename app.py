import streamlit as st
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os

# Set page config
st.set_page_config(page_title="Digit Classifier", page_icon="🔢", layout="wide")

# Load models with error handling
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        "SVM": "best_svm_model.pkl",
        "KNN": "best_knn_model.pkl", 
        "Decision Tree": "best_decision_tree_model.pkl"
    }
    
    for name, file_path in model_files.items():
        try:
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    models[name] = pickle.load(f)
            else:
                st.error(f"Model file {file_path} not found!")
        except Exception as e:
            st.error(f"Error loading {name} model: {str(e)}")
    
    return models

# Load scaler
@st.cache_resource
def load_scaler():
    try:
        if os.path.exists("scaler.pkl"):
            with open("scaler.pkl", "rb") as f:
                return pickle.load(f)
        else:
            st.error("Scaler file not found!")
            return None
    except Exception as e:
        st.error(f"Error loading scaler: {str(e)}")
        return None

# Image preprocessing function
def preprocess_image(image):
    """Convert image to format expected by the model"""
    # Convert to grayscale if not already
    if image.mode != 'L':
        image = image.convert('L')
    
    img_array = np.array(image)
    
    # Resize to 8x8 to match training data format
    resized = cv2.resize(img_array, (8, 8), interpolation=cv2.INTER_AREA)
    
    # Flatten and normalize to match digit dataset (0-16 scale)
    flat = resized.flatten().astype(np.float64)
    scaled_pixel = (flat / 255.0) * 16.0
    
    return resized, scaled_pixel.reshape(1, -1)

# Load models and scaler
models = load_models()
scaler = load_scaler()

# Sidebar
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("---")

# Model selection
if models:
    model_choice = st.sidebar.selectbox("Choose Model", list(models.keys()))
else:
    st.sidebar.error("No models loaded!")
    model_choice = None

# File upload
uploaded_file = st.sidebar.file_uploader(
    "📤 Upload a digit image", 
    type=["jpg", "jpeg", "png", "bmp"],
    help="Upload a clear image of a handwritten digit (0-9)"
)

# Main interface
st.title("🔢 Digit Classifier")
st.markdown(
    """
    Upload a **handwritten digit image** and let the AI classify it! 
    
    📝 **Tips for best results:**
    - Use clear, handwritten digits
    - Ensure good contrast (dark digit on light background)
    - Center the digit in the image
    """
)

# Processing and prediction
if uploaded_file and models and scaler and model_choice:
    try:
        # Load and display original image
        image = Image.open(uploaded_file)
        
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📤 Original Image")
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess image
        processed_8x8, X_input = preprocess_image(image)
        
        # Scale using original scaler
        X_scaled = scaler.transform(X_input)
        
        # Make prediction
        model = models[model_choice]
        prediction = model.predict(X_scaled)[0]
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(X_scaled)[0]
                confidence = max(probabilities)
            except:
                confidence = None
        else:
            confidence = None
        
        with col2:
            st.subheader("🖼️ Processed Image (8×8)")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(processed_8x8, cmap='gray')
            ax.axis('off')
            ax.set_title("What the model sees")
            st.pyplot(fig)
            plt.close()
        
        # Show prediction results
        st.markdown("---")
        
        # Main prediction display
        st.markdown(
            f"<h2 style='text-align: center;'>🎯 Predicted Digit: <span style='color: green; font-size: 3em;'>{prediction}</span></h2>",
            unsafe_allow_html=True
        )
        
        # Show model info and confidence
        col3, col4 = st.columns(2)
        
        with col3:
            st.success(f"✅ **Model Used**: {model_choice}")
        
        with col4:
            if confidence:
                st.info(f"🎯 **Confidence**: {confidence:.2%}")
        
        # Additional info
        with st.expander("ℹ️ Technical Details"):
            st.write(f"**Original Image Size**: {image.size}")
            st.write(f"**Processed Size**: 8×8 pixels")
            st.write(f"**Pixel Value Range**: 0-16 (normalized)")
            st.write(f"**Model Type**: {model_choice}")
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.info("Please try uploading a different image or check if the image is valid.")

elif not uploaded_file:
    # Show example or instructions
    st.info("👈 Upload a digit image in the sidebar to begin classification.")
    
    # You can add example images or instructions here
    st.markdown("""
    ### How to use:
    1. Select a model from the sidebar
    2. Upload an image of a handwritten digit
    3. View the prediction results!
    
    ### Supported formats:
    - JPG, JPEG, PNG, BMP
    - Any size (will be resized to 8×8)
    - Color or grayscale
    """)

else:
    st.error("Please ensure all model files and scaler are available in the current directory.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Built with Streamlit 🚀</p>",
    unsafe_allow_html=True
)
