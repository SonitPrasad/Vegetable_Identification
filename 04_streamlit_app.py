import streamlit as st
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import tempfile


# =========================
# CONFIG
# =========================

MODEL_PATH = "best.pt"

CONFIDENCE_THRESHOLD = 0.80


# =========================
# LABEL CONVERSION
# =========================

def convert_label(class_name):
    """
    Converts labels like:
    okra_l -> Okra, Large
    potato_m -> Potato, Medium
    tomato_s -> Tomato, Small
    """

    parts = class_name.split("_")

    if len(parts) != 2:
        return class_name.capitalize(), "Unknown"

    vegetable = parts[0].capitalize()
    size_code = parts[1].lower()

    size_map = {
        "l": "Large",
        "m": "Medium",
        "s": "Small"
    }

    cut_size = size_map.get(size_code, "Unknown")

    return vegetable, cut_size


# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():
    model_path = Path(MODEL_PATH)

    if not model_path.exists():
        st.error(f"Model not found at: {MODEL_PATH}")
        st.stop()

    model = YOLO(MODEL_PATH)
    return model


# =========================
# PREDICT FUNCTION
# =========================

def predict_uploaded_image(model, image_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(image_file.getvalue())
        temp_image_path = temp_file.name

    results = model.predict(
        source=temp_image_path,
        device="cpu",
        verbose=False
    )

    result = results[0]

    top_class_id = int(result.probs.top1)
    confidence = float(result.probs.top1conf)

    class_name = model.names[top_class_id]

    return class_name, confidence


# =========================
# STREAMLIT UI
# =========================

st.set_page_config(
    page_title="Vegetable Cut Size Classifier",
    page_icon="🥦",
    layout="centered"
)

st.title("Vegetable + Cut Size Classifier")
st.write("Upload an image to classify vegetable type and cut size.")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload vegetable image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict"):
        class_name, confidence = predict_uploaded_image(model, uploaded_file)

        st.subheader("Prediction Result")

        if confidence < CONFIDENCE_THRESHOLD:
            st.warning("Unknown / Not confident")
            st.write(f"Detected Class: `{class_name}`")
            st.write(f"Confidence: `{confidence * 100:.2f}%`")

        else:
            vegetable, cut_size = convert_label(class_name)

            st.success("Prediction successful")
            st.write(f"Vegetable: **{vegetable}**")
            st.write(f"Cut Size: **{cut_size}**")
            st.write(f"Confidence: **{confidence * 100:.2f}%**")
            st.write(f"Raw Class: `{class_name}`")