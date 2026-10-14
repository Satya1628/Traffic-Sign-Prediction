import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# -------------------- Configuration --------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "traffic_sign_cnn.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"
IMAGE_SIZE = (32, 32)

st.set_page_config(
    page_title="SignSense AI | Traffic Sign Recognition",
    page_icon="🚦",
    layout="centered",
)

# -------------------- Load model and class names --------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def load_class_names():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Supports either a list of names or a dictionary keyed by class ID.
    if isinstance(data, list):
        return [str(item) for item in data]

    if isinstance(data, dict):
        try:
            return [str(data[key]) for key in sorted(data, key=lambda x: int(x))]
        except (ValueError, TypeError):
            return list(data.values())

    raise ValueError("class_names.json must contain a list or dictionary.")

def preprocess_image(image):
    # Match the notebook's expected input: RGB, 32x32, pixel values in [0, 1].
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)

# -------------------- Page layout --------------------
st.title("🚦 SignSense AI")
st.subheader("Traffic Sign Recognition")
st.write(
    "Upload a traffic-sign image. The trained CNN will predict its class "
    "and show the most likely alternatives."
)

if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
    st.error(
        "Required files are missing. Put traffic_sign_cnn.keras and "
        "class_names.json in the same folder as app.py."
    )
    st.stop()

try:
    model = load_model()
    class_names = load_class_names()
except Exception as error:
    st.error(f"Could not load the model or class names: {error}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a traffic-sign image",
    type=["jpg", "jpeg", "png", "bmp"],
)

if uploaded_file is not None:
    try:
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption="Uploaded image", use_container_width=True)

        if st.button("Recognize traffic sign", type="primary"):
            input_array = preprocess_image(original_image)
            predictions = model.predict(input_array, verbose=0)[0]

            if len(predictions) != len(class_names):
                st.error(
                    f"The model returned {len(predictions)} outputs, but "
                    f"class_names.json contains {len(class_names)} names. "
                    "Check that this JSON file came from the same notebook run."
                )
                st.stop()

            predicted_id = int(np.argmax(predictions))
            confidence = float(predictions[predicted_id])

            st.success("Prediction complete")
            st.metric("Predicted traffic sign", class_names[predicted_id])
            st.metric("Model confidence", f"{confidence:.2%}")

            st.markdown("### Top 3 predictions")
            top_indices = np.argsort(predictions)[::-1][:3]
            for class_id in top_indices:
                st.write(f"**{class_names[int(class_id)]}** — {predictions[class_id]:.2%}")
                st.progress(float(np.clip(predictions[class_id], 0.0, 1.0)))

            st.caption(
                "Note: confidence is the model's softmax score, not a guarantee. "
                "The model was trained on cropped 32×32 traffic-sign images. "
                "For best results, upload a clear image where one sign fills most "
                "of the frame."
            )
    except Exception as error:
        st.error(f"Could not process this image: {error}")
else:
    st.info("Upload a clear image of one traffic sign to begin.")

st.divider()
st.caption("Deep Learning project • TensorFlow/Keras CNN • 43 traffic-sign classes")
