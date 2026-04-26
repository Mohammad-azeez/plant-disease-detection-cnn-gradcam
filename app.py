import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import gdown
from PIL import Image
import matplotlib.pyplot as plt
import shap
import time

# --- CONFIGURATION ---
st.set_page_config(
    page_title="PlantDocs | Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        color: #e8f5e9;
        transform: scale(1.02);
    }
    .prediction-card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 5px solid #2e7d32;
    }
    .confidence-meter {
        font-size: 24px;
        font-weight: bold;
        color: #2e7d32;
    }
    .title-text {
        color: #1b5e20;
        font-family: 'Inter', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# --- CONSTANTS ---
MODEL_PATH = "best_model.h5"
GDRIVE_ID = "1M5hAwR3CJCELEVVzGeELpHtQZY6gk6PW"
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    'Pepper (Bell) - Bacterial Spot',
    'Pepper (Bell) - Healthy',
    'Potato - Early Blight',
    'Potato - Late Blight',
    'Potato - Healthy',
    'Tomato - Bacterial Spot',
    'Tomato - Early Blight',
    'Tomato - Late Blight',
    'Tomato - Leaf Mold',
    'Tomato - Septoria Leaf Spot',
    'Tomato - Spider Mites (Two-spotted)',
    'Tomato - Target Spot',
    'Tomato - Yellow Leaf Curl Virus',
    'Tomato - Mosaic Virus',
    'Tomato - Healthy'
]

# --- UTILS ---
@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive... This might take a moment."):
            url = f'https://drive.google.com/uc?id={GDRIVE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False)
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    img = image.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def get_gradcam(model, img_array, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(original_img, heatmap, intensity=0.5, res=224):
    heatmap = cv2.resize(heatmap, (res, res))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    img = np.array(original_img.resize((res, res)))
    superimposed_img = heatmap * intensity + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img

# --- APP LAYOUT ---
def main():
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=100)
    st.sidebar.title("PlantDocs 🌿")
    st.sidebar.markdown("""
    Protect your crops with AI-powered disease detection. 
    
    **How it works:**
    1. Upload a clear photo of a plant leaf.
    2. Wait for the neural network to analyze.
    3. View the prediction and visual explanations.
    """)
    
    st.sidebar.divider()
    st.sidebar.info("Model: Custom CNN Architecture\n\nXAI: Grad-CAM & SHAP")

    st.markdown("<h1 class='title-text'>🌿 Plant Disease Intelligence</h1>", unsafe_allow_html=True)
    st.write("Upload an image of a leaf to identify potential diseases and see why the model made that decision.")

    model = load_trained_model()
    
    if model:
        # Find last conv layer for Grad-CAM
        conv_layers = [layer.name for layer in model.layers if 'conv' in layer.name]
        last_conv_layer = conv_layers[-1] if conv_layers else None

        uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])
            
            image = Image.open(uploaded_file).convert('RGB')
            
            with col1:
                st.subheader("Uploaded Image")
                st.image(image, use_container_width=True)
            
            if st.button("🔍 Analyze Leaf"):
                with st.spinner("Analyzing..."):
                    # Preprocess and Predict
                    img_array = preprocess_image(image)
                    preds = model.predict(img_array)
                    
                    st.session_state.analysis_done = True
                    st.session_state.img_array = img_array
                    st.session_state.preds = preds
                    st.session_state.class_idx = np.argmax(preds[0])
                    st.session_state.confidence = preds[0][st.session_state.class_idx] * 100
                    
                    if last_conv_layer:
                        st.session_state.heatmap = get_gradcam(model, img_array, last_conv_layer)

            if st.session_state.get("analysis_done"):
                with col2:
                    st.subheader("Analysis Results")
                    st.markdown(f"""
                        <div class="prediction-card">
                            <h3>Prediction</h3>
                            <p style="font-size: 22px; color: #1b5e20;"><b>{CLASS_NAMES[st.session_state.class_idx]}</b></p>
                            <hr>
                            <h3>Confidence</h3>
                            <p class="confidence-meter">{st.session_state.confidence:.2f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if st.session_state.confidence < 60:
                        st.warning("⚠️ Low confidence. Please ensure the leaf is centered and well-lit.")
                
                st.divider()
                
                # --- EXPLAINABILITY ---
                st.header("🔬 Explainable AI Insights")
                tab1, tab2, tab3 = st.tabs(["🎯 Grad-CAM", "📉 SHAP", "🖼️ Feature Maps"])
                
                with tab1:
                    st.subheader("Grad-CAM (Visual Attention)")
                    st.write("Grad-CAM highlights the specific areas of the image that influenced the model's prediction.")
                    if last_conv_layer and "heatmap" in st.session_state:
                        grad_cam_img = display_gradcam(image, st.session_state.heatmap)
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            st.image(grad_cam_img, caption="Grad-CAM Visualization", use_container_width=True)
                        with c2:
                            st.write("### How to read this:")
                            st.write("- **Red areas**: High influence on prediction.")
                            st.write("- **Blue areas**: Low influence.")
                            st.write("- If the red areas align with visible lesions or spots, the model is focusing on the correct features!")
                    else:
                        st.error("Could not find a convolutional layer for Grad-CAM.")

                with tab2:
                    st.subheader("SHAP (Feature Importance)")
                    st.write("SHAP (SHapley Additive exPlanations) provides a mathematical breakdown of pixel contributions.")
                    if st.button("🚀 Generate SHAP Visualization"):
                        try:
                            with st.spinner("Generating SHAP values (this may take a minute)..."):
                                background = st.session_state.img_array * 0.9 
                                explainer = shap.GradientExplainer(model, background)
                                shap_values = explainer.shap_values(st.session_state.img_array)
                                
                                plt.figure()
                                shap.image_plot(shap_values, st.session_state.img_array, show=False)
                                st.pyplot(plt.gcf(), clear_figure=True)
                        except Exception as e:
                            st.info("SHAP visualization is currently unavailable or taking too long.")
                            st.write(f"Error details: {e}")

                with tab3:
                    st.subheader("Internal Feature Maps")
                    st.write("Feature maps show what the neural network 'sees' at different layers of abstraction.")
                    
                    selected_layer = st.selectbox("Select Layer to Visualize", conv_layers[::-1])
                    
                    if selected_layer:
                        with st.spinner(f"Extracting features from {selected_layer}..."):
                            activation_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(selected_layer).output)
                            activations = activation_model.predict(st.session_state.img_array)
                            
                            num_filters = min(12, activations.shape[-1])
                            cols = st.columns(4)
                            for i in range(num_filters):
                                with cols[i % 4]:
                                    fig, ax = plt.subplots()
                                    ax.imshow(activations[0, :, :, i], cmap='viridis')
                                    ax.axis('off')
                                    st.pyplot(fig)
                                    plt.close(fig)

    else:
        st.error("Failed to initialize the AI model. Please check the logs.")

if __name__ == "__main__":
    main()
