import streamlit as st
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="VisionEval",
    page_icon="🔬",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🔬 VisionEval")
st.subheader("Multimodal AI Model Evaluation Platform")

st.write(
    "Compare multimodal AI models and analyze their performance "
    "across different vision-language tasks."
)


# =========================================================
# CLIP LABELS
# =========================================================

LABELS = [
    "a photo of a dog",
    "a photo of a cat",
    "a photo of a car",
    "a photo of a person",
    "a photo of a bird",
    "a photo of a building",
    "a photo of a computer",
    "a photo of a padlock"
]


# =========================================================
# LOAD CLIP MODEL
# =========================================================

@st.cache_resource
def load_clip_model():

    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    return model, processor


# =========================================================
# IMAGE UPLOAD
# =========================================================

st.divider()

st.header("📷 Upload an Image")

uploaded_image = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# DISPLAY IMAGE
# =========================================================

image = None

if uploaded_image is not None:

    image = Image.open(uploaded_image).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=500
    )

    st.success("Image uploaded successfully! ✅")


# =========================================================
# TASK SELECTION
# =========================================================

st.divider()

st.header("🎯 Select Evaluation Task")

task = st.selectbox(
    "Choose a task",
    [
        "Image Classification",
        "Image Captioning",
        "Visual Question Answering"
    ]
)

st.write("Selected task:", task)


# =========================================================
# IMAGE CLASSIFICATION
# =========================================================

if image is not None and task == "Image Classification":

    st.divider()

    st.header("🤖 CLIP Evaluation")

    st.write(
        "CLIP will compare the uploaded image with the candidate "
        "text descriptions and select the best matching class."
    )

    if st.button("Run CLIP Classification 🚀"):

        with st.spinner("Loading CLIP model..."):

            model, processor = load_clip_model()

        with st.spinner("Running image classification..."):

            inputs = processor(
                text=LABELS,
                images=image,
                return_tensors="pt",
                padding=True
            )

            with torch.no_grad():

                outputs = model(**inputs)

                logits_per_image = outputs.logits_per_image

                probabilities = logits_per_image.softmax(
                    dim=1
                )[0]

        best_index = probabilities.argmax().item()

        prediction = LABELS[best_index]

        confidence = (
            probabilities[best_index].item() * 100
        )

        # Save prediction so it remains available
        # after Streamlit reruns

        st.session_state["prediction"] = prediction

        st.session_state["confidence"] = confidence

        st.session_state["probabilities"] = {
            LABELS[i]: probabilities[i].item() * 100
            for i in range(len(LABELS))
        }

        st.success("CLIP evaluation completed! ✅")

        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        st.subheader("🔮 Prediction")

        predicted_class = prediction.replace(
            "a photo of a ",
            ""
        ).strip()

        st.write(
            f"**Predicted Class:** "
            f"{predicted_class.title()}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        # -------------------------------------------------
        # PROBABILITIES
        # -------------------------------------------------

        st.subheader("📊 Class Probabilities")

        probability_data = {
            label.replace(
                "a photo of a ",
                ""
            ).title(): f"{score:.2f}%"
            for label, score
            in st.session_state["probabilities"].items()
        }

        st.json(probability_data)


# =========================================================
# GROUND TRUTH EVALUATION
# =========================================================

if (
    task == "Image Classification"
    and "prediction" in st.session_state
):

    st.divider()

    st.header("🎯 Ground Truth Evaluation")

    ground_truth = st.selectbox(
        "Select the correct class (Ground Truth):",
        LABELS
    )

    if st.button("Evaluate Prediction 📊"):

        predicted_class = (
            st.session_state["prediction"]
            .replace("a photo of a ", "")
            .strip()
        )

        true_class = (
            ground_truth
            .replace("a photo of a ", "")
            .strip()
        )

        is_correct = (
            predicted_class == true_class
        )

        # Save evaluation result

        st.session_state["ground_truth"] = true_class

        st.session_state["is_correct"] = is_correct

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        if is_correct:

            st.success(
                "✅ Correct Prediction!"
            )

            accuracy = 100

        else:

            st.error(
                "❌ Incorrect Prediction"
            )

            accuracy = 0

        st.metric(
            "Accuracy",
            f"{accuracy}%"
        )

        st.write(
            f"**Ground Truth:** "
            f"{true_class.title()}"
        )

        st.write(
            f"**CLIP Prediction:** "
            f"{predicted_class.title()}"
        )


# =========================================================
# RESULTS DASHBOARD
# =========================================================

if (
    "prediction" in st.session_state
    and "ground_truth" in st.session_state
):

    st.divider()

    st.header("📊 CLIP Results Dashboard")

    prediction_clean = (
        st.session_state["prediction"]
        .replace("a photo of a ", "")
        .strip()
    )

    ground_truth_clean = (
        st.session_state["ground_truth"]
    )

    is_correct = (
        st.session_state["is_correct"]
    )

    confidence = (
        st.session_state["confidence"]
    )

    # -------------------------------------------------
    # THREE METRICS
    # -------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Prediction",
            prediction_clean.title()
        )

    with col2:

        st.metric(
            "Ground Truth",
            ground_truth_clean.title()
        )

    with col3:

        result = (
            "Correct ✅"
            if is_correct
            else "Wrong ❌"
        )

        st.metric(
            "Result",
            result
        )

    # -------------------------------------------------
    # CONFIDENCE
    # -------------------------------------------------

    st.metric(
        "Model Confidence",
        f"{confidence:.2f}%"
    )

    # -------------------------------------------------
    # INTERPRETATION
    # -------------------------------------------------

    st.subheader("📝 Model Interpretation")

    if is_correct:

        st.success(
            f"CLIP correctly classified the image as "
            f"**{prediction_clean.title()}**."
        )

    else:

        st.warning(
            f"CLIP predicted **{prediction_clean.title()}**, "
            f"but the ground truth was "
            f"**{ground_truth_clean.title()}**."
        )

    st.info(
        "This dashboard summarizes the CLIP prediction, "
        "ground truth, confidence, and evaluation result."
    )


# =========================================================
# OTHER TASKS
# =========================================================

if task == "Image Captioning":

    st.divider()

    st.info(
        "🚧 BLIP image captioning will be added in the next version."
    )


if task == "Visual Question Answering":

    st.divider()

    st.info(
        "🚧 LLaVA and Gemini VQA evaluation will be added "
        "in the next version."
    )
