import base64
import io
from datetime import datetime

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import (
    CLIPModel,
    CLIPProcessor,
    BlipForConditionalGeneration,
    BlipProcessor,
)
from huggingface_hub import InferenceClient
from google import genai


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="VisionEval | Multimodal AI Evaluation",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 VisionEval")
st.caption(
    "Evaluate and compare multimodal AI models for image classification, "
    "image captioning, and visual question answering."
)

st.markdown(
    """
    **Tasks available**
    - **Image classification:** CLIP
    - **Image captioning:** BLIP
    - **Visual Question Answering (VQA):** Gemini and Hugging Face vision models
    - **Evaluation:** confidence, accuracy, BLEU, exact match, and CSV downloads
    """
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading CLIP model...")
def load_clip():
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    return processor, model


@st.cache_resource(show_spinner="Loading BLIP captioning model...")
def load_blip():
    model_name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    return processor, model


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def get_secret(secret_name, default=""):
    """Read a Streamlit secret safely."""
    try:
        return str(st.secrets.get(secret_name, default)).strip()
    except Exception:
        return default


def image_to_data_url(image: Image.Image) -> str:
    """Convert a PIL image into a JPEG data URL."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def normalize_answer(answer: str) -> str:
    """Normalize answers for a basic exact-match comparison."""
    return " ".join(str(answer).strip().lower().split()).strip(" .,!?:;")


def make_csv_download(dataframe: pd.DataFrame, filename: str, label: str):
    csv_data = dataframe.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label=label,
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def show_api_error(provider_name: str, error: Exception):
    st.error(f"{provider_name} request failed.")
    st.code(str(error))
    st.caption(
        "Check your API key, model access, provider availability, and "
        "the model identifier. Never share your API keys."
    )


def calculate_bleu(reference: str, candidate: str) -> float:
    """Calculate sentence-level BLEU with smoothing."""
    reference_tokens = reference.lower().strip().split()
    candidate_tokens = candidate.lower().strip().split()

    if not reference_tokens or not candidate_tokens:
        return 0.0

    smoothing = SmoothingFunction().method1
    return float(
        sentence_bleu(
            [reference_tokens],
            candidate_tokens,
            smoothing_function=smoothing,
        )
    )


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.header("About VisionEval")
    st.write("A multimodal AI model evaluation dashboard.")
    st.write(f"**Runtime device:** `{DEVICE}`")
    st.divider()
    st.subheader("API configuration")
    st.write(
        "Add GEMINI_API_KEY and HF_TOKEN in Streamlit Secrets. "
        "Do not place API keys directly in this file."
    )
    st.caption("CLIP and BLIP are downloaded from Hugging Face.")


# ---------------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------------
st.header("1. Upload an image")

uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["png", "jpg", "jpeg", "webp"],
)

if uploaded_file is None:
    st.info("Upload an image to start evaluating the models.")
    st.stop()

try:
    image = Image.open(uploaded_file).convert("RGB")
except Exception as error:
    st.error(f"Could not open this image: {error}")
    st.stop()

left, right = st.columns(2)

with left:
    st.image(image, caption="Uploaded image", use_container_width=True)

with right:
    st.subheader("Image details")
    st.write(f"**File name:** {uploaded_file.name}")
    st.write(f"**Dimensions:** {image.width} × {image.height}")
    st.write(f"**Format:** {uploaded_file.type or 'Unknown'}")

st.divider()


# ---------------------------------------------------------
# TASK SELECTOR
# ---------------------------------------------------------
st.header("2. Select an evaluation task")

task = st.radio(
    "Choose a task",
    [
        "Image Classification (CLIP)",
        "Image Captioning (BLIP)",
        "Visual Question Answering (VQA)",
    ],
    horizontal=True,
)


# =========================================================
# TASK 1: CLIP IMAGE CLASSIFICATION
# =========================================================
if task == "Image Classification (CLIP)":

    st.subheader("CLIP — Image Classification")

    st.write(
        "Enter candidate labels separated by commas. CLIP compares "
        "the image with each label and returns similarity-based probabilities."
    )

    labels_text = st.text_input(
        "Candidate labels",
        value="cat, dog, bird, car, person",
        help="Enter at least two labels, separated by commas.",
    )

    labels = list(
        dict.fromkeys(
            label.strip()
            for label in labels_text.split(",")
            if label.strip()
        )
    )

    ground_truth = st.selectbox(
        "Ground-truth label (optional)",
        ["Not provided"] + labels,
        index=0,
    )

    if st.button(
        "Run CLIP classification",
        type="primary",
        use_container_width=True,
    ):

        if len(labels) < 2:
            st.warning("Please enter at least two different candidate labels.")

        else:
            try:
                processor, model = load_clip()

                inputs = processor(
                    text=labels,
                    images=image,
                    return_tensors="pt",
                    padding=True,
                )

                inputs = {
    key: value.to(DEVICE)
    for key, value in inputs.items()
}
