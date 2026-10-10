
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


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="VisionEval | Multimodal AI Evaluation",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 VisionEval")
st.caption("Multimodal AI Model Evaluation Dashboard")

st.markdown("""
Evaluate models for:
- Image Classification using CLIP
- Image Captioning using BLIP
- Visual Question Answering using Gemini and Hugging Face
- Confidence, accuracy, BLEU, exact match, and CSV results
""")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
@st.cache_resource(show_spinner="Loading CLIP model...")
def load_clip():
    name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(name)
    model = CLIPModel.from_pretrained(name)
    model.to(DEVICE)
    model.eval()
    return processor, model


@st.cache_resource(show_spinner="Loading BLIP model...")
def load_blip():
    name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(name)
    model = BlipForConditionalGeneration.from_pretrained(name)
    model.to(DEVICE)
    model.eval()
    return processor, model


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def get_secret(name):
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


def image_to_data_url(image):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def normalize_answer(answer):
    return " ".join(
        str(answer).strip().lower().split()
    ).strip(" .,!?:;")


def calculate_bleu(reference, candidate):
    reference_tokens = reference.lower().strip().split()
    candidate_tokens = candidate.lower().strip().split()

    if not reference_tokens or not candidate_tokens:
        return 0.0

    return float(
        sentence_bleu(
            [reference_tokens],
            candidate_tokens,
            smoothing_function=SmoothingFunction().method1,
        )
    )


def download_csv(dataframe, filename, label):
    csv_data = dataframe.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label=label,
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def show_error(provider, error):
    st.error(f"{provider} request failed.")
    st.code(str(error))
    st.caption(
        "Check your API key, model identifier, and provider access. "
        "Never share your API keys."
    )


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.header("About VisionEval")
    st.write("A multimodal AI evaluation dashboard.")
    st.write(f"Runtime device: {DEVICE}")
    st.divider()
    st.write("Required Streamlit secrets:")
    st.code("GEMINI_API_KEY\nHF_TOKEN")


# --------------------------------------------------
# UPLOAD IMAGE
# --------------------------------------------------
st.header("1. Upload an image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg", "webp"],
)

if uploaded_file is None:
    st.info("Upload an image to begin.")
    st.stop()

try:
    image = Image.open(uploaded_file).convert("RGB")
except Exception as error:
    st.error(f"Could not open image: {error}")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.image(image, caption="Uploaded image", use_container_width=True)

with col2:
    st.subheader("Image details")
    st.write(f"File: {uploaded_file.name}")
    st.write(f"Dimensions: {image.width} × {image.height}")
    st.write(f"Format: {uploaded_file.type or 'Unknown'}")

st.divider()


# --------------------------------------------------
# SELECT TASK
# --------------------------------------------------
st.header("2. Select an evaluation task")

task = st.radio(
    "Choose task",
    [
        "Image Classification (CLIP)",
        "Image Captioning (BLIP)",
        "Visual Question Answering (VQA)",
    ],
    horizontal=True,
)


# ==================================================
# TASK 1: CLIP CLASSIFICATION
# ==================================================
if task == "Image Classification (CLIP)":

    st.subheader("CLIP — Image Classification")

    labels_text = st.text_input(
        "Candidate labels separated by commas",
        value="cat, dog, bird, car, person",
    )

    labels = list(dict.fromkeys(
        label.strip()
        for label in labels_text.split(",")
        if label.strip()
    ))

    ground_truth = st.selectbox(
        "Ground-truth label (optional)",
        ["Not provided"] + labels,
    )

    if st.button(
        "Run CLIP classification",
        type="primary",
        use_container_width=True,
    ):

        if len(labels) < 2:
            st.warning("Enter at least two different labels.")

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

                with torch.inference_mode():
                    outputs = model(**inputs)
                    probabilities = outputs.logits_per_image.softmax(dim=1)[0]

                    predicted_index = int(
                        torch.argmax(probabilities).item()
                    )

                    predicted_label = labels[predicted_index]
                    confidence = float(
                        probabilities[predicted_index].item()
                    )

                accuracy = (
                    int(
                        predicted_label.lower()
                        == ground_truth.lower()
                    )
                    if ground_truth != "Not provided"
                    else None
                )

                st.success(f"Predicted label: {predicted_label}")

                m1, m2, m3 = st.columns(3)

                m1.metric("Prediction", predicted_label)
                m2.metric("Confidence", f"{confidence:.2%}")
                m3.metric(
                    "Accuracy for this image",
                    f"{accuracy * 100:.0f}%"
                    if accuracy is not None
                    else "N/A",
                )

                probability_df = pd.DataFrame({
                    "Label": labels,
                    "Probability": [
                        float(value)
                        for value in probabilities.cpu()
                    ],
                }).sort_values(
                    "Probability",
                    ascending=False,
                )

                st.subheader("Class probabilities")
                st.bar_chart(
                    probability_df.set_index("Label")["Probability"]
                )

                result_df = pd.DataFrame([{
                    "timestamp": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    "task": "Image Classification",
                    "image_name": uploaded_file.name,
                    "model": "openai/clip-vit-base-patch32",
                    "candidate_labels": ", ".join(labels),
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "ground_truth": (
                        ground_truth
                        if ground_truth != "Not provided"
                        else ""
                    ),
                    "accuracy": (
                        accuracy
                        if accuracy is not None
                        else ""
                    ),
                }])

                st.subheader("Evaluation results")
                st.dataframe(result_df, use_container_width=True)

                download_csv(
                    result_df,
                    "visioneval_clip_results.csv",
                    "Download CLIP results (CSV)",
                )

                with st.expander("View all probabilities"):
                    st.dataframe(
                        probability_df,
                        use_container_width=True,
                    )

            except Exception as error:
                show_error("CLIP", error)


# ==================================================
# TASK 2: BLIP IMAGE CAPTIONING
# ==================================================
elif task == "Image Captioning (BLIP)":

    st.subheader("BLIP — Image Captioning")

    reference_caption = st.text_area(
        "Reference caption (optional)",
        placeholder="Example: A dog is running in a park.",
    )

    if st.button(
        "Generate image caption",
        type="primary",
        use_container_width=True,
    ):

        try:
            processor, model = load_blip()

            inputs = processor(
                images=image,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(DEVICE)
                for key, value in inputs.items()
            }

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=60,
                    num_beams=4,
                )

            caption = processor.decode(
                output_ids[0],
                skip_special_tokens=True,
            )

            bleu_score = (
                calculate_bleu(reference_caption, caption)
                if reference_caption.strip()
                else None
            )

            st.success("Caption generated successfully.")
            st.subheader("Generated caption")
            st.write(caption)

            c1, c2 = st.columns(2)

            c1.metric("Word count", len(caption.split()))

            c2.metric(
                "Sentence BLEU",
                f"{bleu_score:.4f}"
                if bleu_score is not None
                else "N/A",
            )

            if not reference_caption.strip():
                st.info(
                    "Provide a reference caption to calculate BLEU."
                )

            result_df = pd.DataFrame([{
                "timestamp": datetime.now().isoformat(
                    timespec="seconds"
                ),
                "task": "Image Captioning",
                "image_name": uploaded_file.name,
                "model": "Salesforce/blip-image-captioning-base",
                "generated_caption": caption,
                "reference_caption": reference_caption.strip(),
                "bleu_score": (
                    bleu_score
                    if bleu_score is not None
                    else ""
                ),
            }])

            st.subheader("Evaluation results")
            st.dataframe(result_df, use_container_width=True)

            download_csv(
                result_df,
                "visioneval_blip_results.csv",
                "Download captioning results (CSV)",
            )

        except Exception as error:
            show_error("BLIP", error)


# ==================================================
# TASK 3: VISUAL QUESTION ANSWERING
# ==================================================
else:

    st.subheader("Visual Question Answering")

    question = st.text_input(
        "Question about the image",
        value="What is shown in this image?",
        max_chars=500,
    )

    reference_answer = st.text_input(
        "Reference answer (optional)",
        placeholder="Enter the expected answer.",
        max_chars=300,
    )

    col1, col2 = st.columns(2)

    with col1:
        run_gemini = st.checkbox("Run Gemini", value=True)

    with col2:
        run_huggingface = st.checkbox(
            "Run Hugging Face vision model",
            value=True,
        )

    hf_model = st.text_input(
        "Hugging Face vision model ID",
        value="Qwen/Qwen2.5-VL-3B-Instruct",
        help=(
            "Use a vision-language model available through your "
            "enabled Hugging Face Inference Providers."
        ),
    )

    if st.button(
        "Run VQA evaluation",
        type="primary",
        use_container_width=True,
    ):

        if not question.strip():
            st.warning("Enter a question.")

        elif not run_gemini and not run_huggingface:
            st.warning("Select at least one model.")

        else:
            results = []

            # ------------------------------------------
            # GEMINI
            # ------------------------------------------
            if run_gemini:

                gemini_key = get_secret("GEMINI_API_KEY")

                if not gemini_key:
                    st.error(
                        "GEMINI_API_KEY is missing from Streamlit Secrets."
                    )

                else:
                    with st.spinner("Asking Gemini..."):

                        try:
                            client = genai.Client(
                                api_key=gemini_key
                            )

                            response = client.models.generate_content(
                                model="gemini-3.8-flash",
                                contents=[question.strip(), image],
                            )

                            answer = (response.text or "").strip()

                            if not answer:
                                answer = "Empty response from Gemini."

                            exact_match = (
                                normalize_answer(answer)
                                == normalize_answer(reference_answer)
                                if reference_answer.strip()
                                else None
                            )

                            results.append({
                                "timestamp": datetime.now().isoformat(
                                    timespec="seconds"
                                ),
                                "task": "VQA",
                                "image_name": uploaded_file.name,
                                "model": "gemini-3.8-flash",
                                "question": question.strip(),
                                "answer": answer,
                                "reference_answer": reference_answer.strip(),
                                "exact_match": (
                                    exact_match
                                    if exact_match is not None
                                    else ""
                                ),
                                "status": "Success",
                            })

                        except Exception as error:
                            show_error("Gemini", error)

            # ------------------------------------------
            # HUGGING FACE VISION MODEL
            # ------------------------------------------
            if run_huggingface:

                hf_token = get_secret("HF_TOKEN")

                if not hf_token:
                    st.error(
                        "HF_TOKEN is missing from Streamlit Secrets."
                    )

                elif not hf_model.strip():
                    st.error("Enter a Hugging Face model ID.")

                else:
                    with st.spinner(
                        "Asking Hugging Face vision model..."
                    ):

                        try:
                            hf_client = InferenceClient(
                                token=hf_token,
                                timeout=120,
                            )

                            image_data_url = image_to_data_url(image)

                            response = hf_client.chat.completions.create(
                                model=hf_model.strip(),
                                messages=[
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": question.strip(),
                                            },
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": image_data_url,
                                                },
                                            },
                                        ],
                                    }
                                ],
                                max_tokens=150,
                            )

                            answer = (
                                response.choices[0].message.content or ""
                            ).strip()

                            if not answer:
                                answer = (
                                    "Empty response from Hugging Face model."
                                )

                            exact_match = (
                                normalize_answer(answer)
                                == normalize_answer(reference_answer)
                                if reference_answer.strip()
                                else None
                            )

                            results.append({
                                "timestamp": datetime.now().isoformat(
                                    timespec="seconds"
                                ),
                                "task": "VQA",
                                "image_name": uploaded_file.name,
                                "model": hf_model.strip(),
                                "question": question.strip(),
                                "answer": answer,
                                "reference_answer": reference_answer.strip(),
                                "exact_match": (
                                    exact_match
                                    if exact_match is not None
                                    else ""
                                ),
                                "status": "Success",
                            })

                        except Exception as error:
                            show_error(
                                "Hugging Face vision model",
                                error,
                            )

                            st.warning(
                                "If the error is model_not_supported, "
                                "the model is not available through an "
                                "enabled provider for your account. "
                                "A token alone cannot enable an unsupported "
                                "model. Choose a compatible vision model "
                                "and its exact model ID."
                            )

            # ------------------------------------------
            # DISPLAY RESULTS
            # ------------------------------------------
            if results:

                st.divider()
                st.subheader("Model answers")

                for result in results:

                    with st.container(border=True):

                        st.markdown(f"**{result['model']}**")
                        st.write(result["answer"])

                        if result["exact_match"] == "":
                            st.caption(
                                "Exact match: N/A — no reference answer"
                            )
                        else:
                            st.caption(
                                "Exact match: "
                                + (
                                    "Yes"
                                    if result["exact_match"]
                                    else "No"
                                )
                            )

                result_df = pd.DataFrame(results)

                st.subheader("VQA comparison table")
                st.dataframe(result_df, use_container_width=True)

                download_csv(
                    result_df,
                    "visioneval_vqa_results.csv",
                    "Download VQA results (CSV)",
                )

                if len(results) >= 2:

                    st.subheader("Answer comparison")

                    comparison_df = pd.DataFrame([
                        {
                            "Model": row["model"],
                            "Answer": row["answer"],
                            "Exact match": (
                                "N/A"
                                if row["exact_match"] == ""
                                else (
                                    "Yes"
                                    if row["exact_match"]
                                    else "No"
                                )
                            ),
                        }
                        for row in results
                    ])

                    st.dataframe(
                        comparison_df,
                        use_container_width=True,
                    )

                st.caption(
                    "Exact match compares normalized answer strings. "
                    "It does not measure semantic similarity or fully "
                    "evaluate VQA correctness."
                )

            else:
                st.info(
                    "No model returned a successful answer. "
                    "Review the errors and check API keys and model access."
                )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "VisionEval is an educational prototype. Results depend on the "
    "image, labels, reference answers, model versions, and providers."
)
