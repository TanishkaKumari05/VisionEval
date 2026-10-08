import streamlit as st
import torch
import pandas as pd

from PIL import Image
from transformers import (
    CLIPModel,
    CLIPProcessor,
    BlipProcessor,
    BlipForConditionalGeneration
)

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="VisionEval",
    page_icon="👁️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("👁️ VisionEval")
st.subheader("Multimodal AI Model Evaluation & Error Analysis Platform")

st.write(
    """
    VisionEval evaluates multimodal AI models using task-specific
    evaluation metrics and provides an easy-to-understand comparison
    of model performance.
    """
)

st.divider()


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Evaluation Settings")

st.sidebar.write(f"**Device:** `{device}`")

task = st.sidebar.selectbox(
    "Select Evaluation Task",
    [
        "Image Classification",
        "Image Captioning",
        "Visual Question Answering"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    """
    **VisionEval Models**

    🔵 CLIP → Image Classification

    🟢 BLIP → Image Captioning

    🟣 LLaVA → VQA

    🔴 Gemini → VQA
    """
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.header("📷 Upload an Image")

uploaded_image = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_image is not None:

    image = Image.open(uploaded_image).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=400
    )

    st.divider()


# ============================================================
# CLIP MODEL
# ============================================================

@st.cache_resource
def load_clip_model():

    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    model.to(device)
    model.eval()

    return processor, model


# ============================================================
# BLIP MODEL
# ============================================================

@st.cache_resource
def load_blip_model():

    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model.to(device)
    model.eval()

    return processor, model


# ============================================================
# CLIP IMAGE CLASSIFICATION
# ============================================================

if task == "Image Classification":

    st.header("🔵 CLIP Image Classification")

    st.write(
        """
        CLIP compares the uploaded image with candidate text labels
        and predicts the label with the highest similarity.
        """
    )

    if uploaded_image is None:

        st.info("👆 Upload an image above to start CLIP evaluation.")

    else:

        # ----------------------------------------------------
        # Candidate Labels
        # ----------------------------------------------------

        labels = [
            "person",
            "dog",
            "cat",
            "car",
            "bird",
            "computer",
            "phone",
            "bicycle",
            "food",
            "building",
            "tree",
            "book",
            "chair",
            "padlock",
            "shoe"
        ]

        ground_truth = st.selectbox(
            "🎯 Select Ground Truth Label",
            labels
        )

        if st.button(
            "🚀 Run CLIP Evaluation",
            type="primary"
        ):

            with st.spinner("Loading CLIP model..."):

                clip_processor, clip_model = load_clip_model()

            with st.spinner("Evaluating image..."):

                text_inputs = [
                    f"a photo of a {label}"
                    for label in labels
                ]

                inputs = clip_processor(
                    text=text_inputs,
                    images=image,
                    return_tensors="pt",
                    padding=True
                )

                inputs = {
                    key: value.to(device)
                    for key, value in inputs.items()
                }

                with torch.no_grad():

                    outputs = clip_model(**inputs)

                    logits_per_image = outputs.logits_per_image

                    probabilities = logits_per_image.softmax(
                        dim=1
                    )[0]

                predicted_index = probabilities.argmax().item()

                prediction = labels[predicted_index]

                confidence = (
                    probabilities[predicted_index].item()
                    * 100
                )

                accuracy = (
                    100
                    if prediction.lower()
                    == ground_truth.lower()
                    else 0
                )

                # ------------------------------------------------
                # Save results
                # ------------------------------------------------

                st.session_state["clip_prediction"] = prediction
                st.session_state["clip_confidence"] = confidence
                st.session_state["clip_ground_truth"] = ground_truth
                st.session_state["clip_accuracy"] = accuracy

                st.session_state["clip_probabilities"] = {
                    labels[i]: probabilities[i].item() * 100
                    for i in range(len(labels))
                }

            st.success("CLIP evaluation completed successfully! ✅")

        # ----------------------------------------------------
        # CLIP RESULTS
        # ----------------------------------------------------

        if "clip_prediction" in st.session_state:

            st.divider()

            st.header("📊 CLIP Results Dashboard")

            prediction = st.session_state["clip_prediction"]
            confidence = st.session_state["clip_confidence"]
            ground_truth = st.session_state["clip_ground_truth"]
            accuracy = st.session_state["clip_accuracy"]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Prediction",
                    prediction.title()
                )

            with col2:
                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            with col3:
                st.metric(
                    "Ground Truth",
                    ground_truth.title()
                )

            with col4:
                st.metric(
                    "Accuracy",
                    f"{accuracy:.0f}%"
                )

            st.divider()

            # ------------------------------------------------
            # Correct / Incorrect
            # ------------------------------------------------

            if prediction.lower() == ground_truth.lower():

                st.success(
                    f"✅ Correct Prediction: {prediction.title()}"
                )

            else:

                st.warning(
                    f"⚠️ CLIP predicted **{prediction.title()}**, "
                    f"but the correct label is **{ground_truth.title()}**."
                )

            st.divider()

            # ------------------------------------------------
            # Probability Distribution
            # ------------------------------------------------

            st.subheader("📈 Class Probability Distribution")

            probability_df = pd.DataFrame(
                {
                    "Class": list(
                        st.session_state[
                            "clip_probabilities"
                        ].keys()
                    ),
                    "Probability (%)": list(
                        st.session_state[
                            "clip_probabilities"
                        ].values()
                    )
                }
            )

            probability_df = probability_df.sort_values(
                "Probability (%)",
                ascending=False
            )

            st.bar_chart(
                probability_df.set_index("Class")
            )

            st.divider()

            # ------------------------------------------------
            # Model Interpretation
            # ------------------------------------------------

            st.subheader("🧠 Model Interpretation")

            if prediction.lower() == ground_truth.lower():

                st.success(
                    f"""
                    CLIP correctly classified the image as
                    **{prediction.title()}** with a confidence of
                    **{confidence:.2f}%**.
                    """
                )

            else:

                st.warning(
                    f"""
                    CLIP predicted **{prediction.title()}**
                    instead of the ground truth
                    **{ground_truth.title()}**.

                    This represents a classification error.
                    """
                )

            st.divider()

            # ------------------------------------------------
            # Evaluation Details
            # ------------------------------------------------

            st.subheader("📋 Detailed Evaluation")

            results_df = pd.DataFrame(
                [
                    {
                        "Model": "CLIP",
                        "Task": "Image Classification",
                        "Prediction": prediction,
                        "Ground Truth": ground_truth,
                        "Confidence (%)": round(
                            confidence,
                            2
                        ),
                        "Accuracy (%)": accuracy
                    }
                ]
            )

            st.dataframe(
                results_df,
                use_container_width=True
            )

            csv = results_df.to_csv(
                index=False
            )

            st.download_button(
                label="⬇️ Download CLIP Results as CSV",
                data=csv,
                file_name="visioneval_clip_results.csv",
                mime="text/csv"
            )

            st.info(
                """
                Accuracy is calculated for the successfully
                evaluated image only.
                """
            )


# ============================================================
# BLIP IMAGE CAPTIONING
# ============================================================

elif task == "Image Captioning":

    st.header("🟢 BLIP Image Captioning")

    st.write(
        """
        BLIP generates a natural-language caption describing
        the uploaded image.
        """
    )

    if uploaded_image is None:

        st.info(
            "👆 Upload an image above to start BLIP captioning."
        )

    else:

        st.subheader("📝 Caption Generation")

        if st.button(
            "🚀 Run BLIP Captioning",
            type="primary"
        ):

            with st.spinner(
                "Loading BLIP model... This may take some time on first run."
            ):

                blip_processor, blip_model = load_blip_model()

            with st.spinner(
                "Generating image caption..."
            ):

                inputs = blip_processor(
                    images=image,
                    return_tensors="pt"
                )

                inputs = {
                    key: value.to(device)
                    for key, value in inputs.items()
                }

                with torch.no_grad():

                    output = blip_model.generate(
                        **inputs,
                        max_new_tokens=30
                    )

                caption = blip_processor.decode(
                    output[0],
                    skip_special_tokens=True
                )

            st.session_state["blip_caption"] = caption

            st.success(
                "BLIP caption generated successfully! ✅"
            )

        # ----------------------------------------------------
        # BLIP RESULTS
        # ----------------------------------------------------

        if "blip_caption" in st.session_state:

            caption = st.session_state["blip_caption"]

            st.divider()

            st.subheader("🤖 Generated Caption")

            st.info(
                caption
            )

            st.divider()

            # ------------------------------------------------
            # Reference Caption
            # ------------------------------------------------

            st.subheader("🎯 Reference Caption")

            reference_caption = st.text_area(
                "Enter the human/reference caption for evaluation:",
                placeholder=(
                    "Example: a dog sitting on a sidewalk "
                    "with a stick in its mouth"
                )
            )

            if reference_caption.strip():

                if st.button(
                    "📊 Calculate BLEU Score",
                    type="primary"
                ):

                    reference_tokens = [
                        reference_caption.lower().split()
                    ]

                    candidate_tokens = (
                        caption.lower().split()
                    )

                    smoothie = (
                        SmoothingFunction().method1
                    )

                    bleu_score = sentence_bleu(
                        reference_tokens,
                        candidate_tokens,
                        smoothing_function=smoothie
                    )

                    bleu_percentage = (
                        bleu_score * 100
                    )

                    st.session_state[
                        "blip_bleu"
                    ] = bleu_score

                    st.session_state[
                        "blip_reference"
                    ] = reference_caption

                    st.session_state[
                        "blip_bleu_percentage"
                    ] = bleu_percentage

            # ------------------------------------------------
            # BLIP Evaluation Dashboard
            # ------------------------------------------------

            if "blip_bleu" in st.session_state:

                st.divider()

                st.header(
                    "📊 BLIP Evaluation Dashboard"
                )

                bleu_score = st.session_state[
                    "blip_bleu"
                ]

                bleu_percentage = st.session_state[
                    "blip_bleu_percentage"
                ]

                reference = st.session_state[
                    "blip_reference"
                ]

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "BLEU Score",
                        f"{bleu_score:.4f}"
                    )

                with col2:

                    st.metric(
                        "BLEU Percentage",
                        f"{bleu_percentage:.2f}%"
                    )

                st.divider()

                st.subheader(
                    "🔍 Caption Comparison"
                )

                comparison_df = pd.DataFrame(
                    [
                        {
                            "Generated Caption": caption,
                            "Reference Caption": reference,
                            "BLEU Score": round(
                                bleu_score,
                                4
                            )
                        }
                    ]
                )

                st.dataframe(
                    comparison_df,
                    use_container_width=True
                )

                st.divider()

                st.subheader(
                    "🧠 BLEU Interpretation"
                )

                if bleu_score >= 0.7:

                    st.success(
                        """
                        High BLEU score.

                        The generated caption has strong
                        word/n-gram overlap with the reference
                        caption.
                        """
                    )

                elif bleu_score >= 0.4:

                    st.warning(
                        """
                        Moderate BLEU score.

                        The generated caption has some
                        similarity with the reference caption.
                        """
                    )

                else:

                    st.error(
                        """
                        Low BLEU score.

                        The generated caption has relatively
                        low word/n-gram overlap with the
                        reference caption.
                        """
                    )

                st.divider()

                # ------------------------------------------------
                # BLIP CSV
                # ------------------------------------------------

                blip_results = pd.DataFrame(
                    [
                        {
                            "Model": "BLIP",
                            "Task": "Image Captioning",
                            "Generated Caption": caption,
                            "Reference Caption": reference,
                            "BLEU Score": round(
                                bleu_score,
                                4
                            ),
                            "BLEU Percentage": round(
                                bleu_percentage,
                                2
                            )
                        }
                    ]
                )

                st.subheader(
                    "📋 Detailed Evaluation"
                )

                st.dataframe(
                    blip_results,
                    use_container_width=True
                )

                blip_csv = blip_results.to_csv(
                    index=False
                )

                st.download_button(
                    label="⬇️ Download BLIP Results as CSV",
                    data=blip_csv,
                    file_name="visioneval_blip_results.csv",
                    mime="text/csv"
                )

                st.info(
                    """
                    BLEU measures similarity between the generated
                    caption and the reference caption using
                    word/n-gram overlap.
                    """
                )


# ============================================================
# VISUAL QUESTION ANSWERING
# ============================================================

elif task == "Visual Question Answering":

    st.header("🟣 Visual Question Answering")

    st.info(
        """
        LLaVA and Gemini VQA evaluation will be added here next.

        Planned evaluation:
        
        Image + Question → Model Answer → Human Answers
        → VQAv2-style Score
        """
    )

    st.markdown(
        """
        ### Planned Models

        **LLaVA**
        - Visual Question Answering
        - VQAv2 validation questions
        - VQAv2-style score

        **Gemini**
        - Visual Question Answering
        - Same VQAv2 questions
        - Same evaluation metric
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "VisionEval — Multimodal AI Model Evaluation & Error Analysis Platform"
)

st.caption(
    "Models: CLIP • BLIP • LLaVA • Gemini | "
    "Metrics: Accuracy • BLEU • VQAv2-style Score"
)
