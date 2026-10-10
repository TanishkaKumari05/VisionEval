import base64
from io import BytesIO

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import CLIPModel, CLIPProcessor, BlipProcessor, BlipForConditionalGeneration
from google import genai
from huggingface_hub import InferenceClient


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(page_title="VisionEval", page_icon="👁️", layout="wide")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# MODEL LOADERS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_clip_model():
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    model.eval()
    return processor, model


@st.cache_resource(show_spinner=False)
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(DEVICE)
    model.eval()
    return processor, model


def normalize_answer(value):
    return " ".join(str(value).lower().strip().split()).rstrip(".,!? ")


def read_secret(name):
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


# ============================================================
# HEADER + SIDEBAR
# ============================================================

st.title("👁️ VisionEval")
st.subheader("Multimodal AI Model Evaluation & Error Analysis Platform")
st.write(
    "Evaluate image classification, image captioning, and visual question "
    "answering with task-specific metrics."
)

st.sidebar.title("⚙️ Evaluation Settings")
st.sidebar.write(f"**Device:** `{DEVICE}`")

task = st.sidebar.selectbox(
    "Select Evaluation Task",
    ["Image Classification", "Image Captioning", "Visual Question Answering"],
)

st.sidebar.divider()
st.sidebar.info(
    "**VisionEval Models**\n\n"
    "🔵 CLIP → Image Classification\n\n"
    "🟢 BLIP → Image Captioning\n\n"
    "🟣 LLaVA → Visual Question Answering\n\n"
    "🔴 Gemini → Visual Question Answering"
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.header("📷 Upload an Image")
uploaded_image = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    key="visioneval_image_uploader",
)

image = None
if uploaded_image is not None:
    try:
        image = Image.open(BytesIO(uploaded_image.getvalue())).convert("RGB")
        st.image(image, caption="Uploaded Image", width=400)
        st.divider()
    except Exception as exc:
        st.error(f"Could not open this image: {exc}")


# ============================================================
# CLIP: IMAGE CLASSIFICATION
# ============================================================

if task == "Image Classification":
    st.header("🔵 CLIP Image Classification")
    st.write(
        "CLIP compares an image with candidate text labels and selects "
        "the label with the highest similarity."
    )

    if image is None:
        st.info("Upload an image above to start CLIP evaluation.")
    else:
        labels = [
            "person", "dog", "cat", "car", "bird", "computer",
            "phone", "bicycle", "food", "building", "tree",
            "book", "chair", "padlock", "shoe",
        ]

        ground_truth = st.selectbox(
            "🎯 Select Ground Truth Label",
            labels,
            key="clip_ground_truth_input",
        )

        if st.button("🚀 Run CLIP Evaluation", type="primary"):
            try:
                with st.spinner("Loading CLIP model..."):
                    processor, model = load_clip_model()

                with st.spinner("Evaluating image..."):
                    texts = [f"a photo of a {label}" for label in labels]
                    inputs = processor(
                        text=texts,
                        images=image,
                        return_tensors="pt",
                        padding=True,
                    )
                    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

                    with torch.no_grad():
                        outputs = model(**inputs)
                        probabilities = outputs.logits_per_image.softmax(dim=1)[0]

                    predicted_index = int(probabilities.argmax().item())
                    prediction = labels[predicted_index]
                    confidence = float(probabilities[predicted_index].item() * 100)
                    accuracy = int(
                        normalize_answer(prediction) == normalize_answer(ground_truth)
                    ) * 100

                st.session_state["clip_result"] = {
                    "prediction": prediction,
                    "confidence": confidence,
                    "ground_truth": ground_truth,
                    "accuracy": accuracy,
                    "probabilities": {
                        labels[i]: float(probabilities[i].item() * 100)
                        for i in range(len(labels))
                    },
                }
                st.success("CLIP evaluation completed! ✅")
            except Exception as exc:
                st.error(f"CLIP evaluation failed: {exc}")

        result = st.session_state.get("clip_result")
        if result:
            st.divider()
            st.header("📊 CLIP Results Dashboard")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Prediction", result["prediction"].title())
            c2.metric("Confidence", f'{result["confidence"]:.2f}%')
            c3.metric("Ground Truth", result["ground_truth"].title())
            c4.metric("Single-image score", f'{result["accuracy"]}%')

            if result["accuracy"] == 100:
                st.success(f'✅ Correct prediction: {result["prediction"].title()}')
            else:
                st.warning(
                    f'CLIP predicted **{result["prediction"].title()}**, but '
                    f'the selected ground truth is **{result["ground_truth"].title()}**.'
                )

            st.subheader("📈 Class Probability Distribution")
            probability_df = pd.DataFrame(
                list(result["probabilities"].items()),
                columns=["Class", "Probability (%)"],
            ).sort_values("Probability (%)", ascending=False)
            st.bar_chart(probability_df.set_index("Class"))

            st.subheader("📋 Detailed Evaluation")
            results_df = pd.DataFrame([{
                "Model": "CLIP",
                "Task": "Image Classification",
                "Prediction": result["prediction"],
                "Ground Truth": result["ground_truth"],
                "Confidence (%)": round(result["confidence"], 2),
                "Single-image score (%)": result["accuracy"],
            }])
            st.dataframe(results_df, use_container_width=True)
            st.download_button(
                "⬇️ Download CLIP Results as CSV",
                data=results_df.to_csv(index=False),
                file_name="visioneval_clip_results.csv",
                mime="text/csv",
            )
            st.caption("Single-image result only; not an official benchmark score.")


# ============================================================
# BLIP: IMAGE CAPTIONING
# ============================================================

elif task == "Image Captioning":
    st.header("🟢 BLIP Image Captioning")
    st.write("BLIP generates a natural-language caption for the uploaded image.")

    if image is None:
        st.info("Upload an image above to start BLIP captioning.")
    else:
        if st.button("🚀 Run BLIP Captioning", type="primary"):
            try:
                with st.spinner("Loading BLIP model..."):
                    processor, model = load_blip_model()

                with st.spinner("Generating caption..."):
                    inputs = processor(images=image, return_tensors="pt")
                    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
                    with torch.no_grad():
                        output = model.generate(**inputs, max_new_tokens=40)
                    caption = processor.decode(output[0], skip_special_tokens=True)

                st.session_state["blip_caption"] = caption
                st.session_state.pop("blip_result", None)
                st.success("BLIP caption generated! ✅")
            except Exception as exc:
                st.error(f"BLIP captioning failed: {exc}")

        caption = st.session_state.get("blip_caption")
        if caption:
            st.subheader("🤖 Generated Caption")
            st.info(caption)

            reference_caption = st.text_area(
                "🎯 Enter a human/reference caption to calculate BLEU:",
                placeholder="Example: a dog sitting on a sidewalk",
                key="blip_reference_input",
            )

            if st.button("📊 Calculate BLEU Score"):
                if not reference_caption.strip():
                    st.warning("Enter a reference caption first.")
                else:
                    score = sentence_bleu(
                        [reference_caption.lower().split()],
                        caption.lower().split(),
                        smoothing_function=SmoothingFunction().method1,
                    )
                    st.session_state["blip_result"] = {
                        "caption": caption,
                        "reference": reference_caption,
                        "bleu": float(score),
                    }

            result = st.session_state.get("blip_result")
            if result:
                st.divider()
                st.header("📊 BLIP Evaluation Dashboard")
                c1, c2 = st.columns(2)
                c1.metric("BLEU score", f'{result["bleu"]:.4f}')
                c2.metric("BLEU (%)", f'{result["bleu"] * 100:.2f}%')

                comparison_df = pd.DataFrame([{
                    "Model": "BLIP",
                    "Task": "Image Captioning",
                    "Generated Caption": result["caption"],
                    "Reference Caption": result["reference"],
                    "BLEU Score": round(result["bleu"], 4),
                }])
                st.dataframe(comparison_df, use_container_width=True)
                st.download_button(
                    "⬇️ Download BLIP Results as CSV",
                    data=comparison_df.to_csv(index=False),
                    file_name="visioneval_blip_results.csv",
                    mime="text/csv",
                )
                st.caption(
                    "BLEU measures n-gram overlap with the reference caption. "
                    "A single reference caption is a demo, not an official COCO score."
                )


# ============================================================
# VQA: LLAVA + GEMINI
# ============================================================

elif task == "Visual Question Answering":
    st.header("🟣 Visual Question Answering")
    st.write(
        "Ask the same question about an image to LLaVA and Gemini, "
        "then compare their answers."
    )

    if image is None:
        st.info("Upload an image above to begin VQA.")
    else:
        question = st.text_input(
            "Ask a question about the image",
            placeholder="Example: What is the person wearing?",
            key="vqa_question_input",
        )
        reference_answer = st.text_input(
            "Reference answer (optional)",
            placeholder="Example: A black jacket",
            key="vqa_reference_input",
        )

        col1, col2 = st.columns(2)
        with col1:
            run_llava = st.checkbox("Run LLaVA", value=True)
        with col2:
            run_gemini = st.checkbox("Run Gemini", value=True)

        if st.button("🚀 Compare AI Models", type="primary"):
            if not question.strip():
                st.warning("Please enter a question.")
            elif not run_llava and not run_gemini:
                st.warning("Select at least one model.")
            else:
                answers = {}

                if run_gemini:
                    st.subheader("🔴 Gemini")
                    gemini_key = read_secret("GEMINI_API_KEY")

                    if not gemini_key:
                        st.error(
                            "GEMINI_API_KEY is missing. Add it in "
                            "Streamlit Cloud → Manage app → Settings → Secrets."
                        )
                    else:
                        try:
                            with st.spinner("Gemini is analysing the image..."):
                                client = genai.Client(api_key=gemini_key)
                                response = client.models.generate_content(
                                    # Change this model name if your API account
                                    # lists a different supported vision model.
                                    model="gemini-2.5-flash",
                                    contents=[question, image],
                                )
                                answer = (response.text or "").strip()
                                if not answer:
                                    raise RuntimeError("Gemini returned an empty response.")
                                answers["Gemini"] = answer
                                st.success(answer)
                        except Exception as exc:
                            st.error(f"Gemini request failed: {exc}")

                if run_llava:
                    st.subheader("🟣 LLaVA")
                    hf_token = read_secret("HF_TOKEN")

                    if not hf_token:
                        st.error(
                            "HF_TOKEN is missing. Add a Hugging Face access token "
                            "in Streamlit Cloud → Manage app → Settings → Secrets."
                        )
                    else:
                        try:
                            with st.spinner("LLaVA is analysing the image..."):
                                image_buffer = BytesIO()
                                image.save(image_buffer, format="JPEG")
                                image_b64 = base64.b64encode(
                                    image_buffer.getvalue()
                                ).decode("utf-8")
                                image_data_url = "data:image/jpeg;base64," + image_b64

                                hf_client = InferenceClient(
                                    token=hf_token,
                                    timeout=120,
                                )
                                response = hf_client.chat.completions.create(
                                    model="llava-hf/llava-1.5-7b-hf",
                                    messages=[{
                                        "role": "user",
                                        "content": [
                                            {"type": "text", "text": question},
                                            {
                                                "type": "image_url",
                                                "image_url": {"url": image_data_url},
                                            },
                                        ],
                                    }],
                                    max_tokens=150,
                                )
                                answer = (response.choices[0].message.content or "").strip()
                                if not answer:
                                    raise RuntimeError("LLaVA returned an empty response.")
                                answers["LLaVA"] = answer
                                st.success(answer)
                        except Exception as exc:
                            st.error(
                                "LLaVA request failed. The model may not be available "
                                "through your Hugging Face inference provider or may "
                                f"require provider access. Details: {exc}"
                            )

                if answers:
                    st.divider()
                    st.subheader("📊 Answer Comparison")
                    comparison_df = pd.DataFrame([
                        {"Model": model_name, "Question": question, "Answer": answer}
                        for model_name, answer in answers.items()
                    ])
                    st.dataframe(comparison_df, use_container_width=True)
                    st.download_button(
                        "⬇️ Download VQA Results as CSV",
                        data=comparison_df.to_csv(index=False),
                        file_name="visioneval_vqa_results.csv",
                        mime="text/csv",
                    )

                    if reference_answer.strip():
                        st.subheader("🎯 Reference Answer Comparison")
                        ref_normalized = normalize_answer(reference_answer)
                        score_rows = []
                        for model_name, answer in answers.items():
                            exact_match = int(
                                normalize_answer(answer) == ref_normalized
                            ) * 100
                            score_rows.append({
                                "Model": model_name,
                                "Reference Answer": reference_answer,
                                "Model Answer": answer,
                                "Exact Match (%)": exact_match,
                            })
                        score_df = pd.DataFrame(score_rows)
                        st.dataframe(score_df, use_container_width=True)
                        st.caption(
                            "Exact-match is a simple demo metric, not the official "
                            "VQAv2 scoring script. Natural-language answers can be "
                            "correct even when their wording differs."
                        )


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("VisionEval — Multimodal AI Model Evaluation & Error Analysis Platform")
st.caption(
    "Models: CLIP • BLIP • LLaVA • Gemini | "
    "Metrics: Accuracy • BLEU • Exact Match (demo)"
)
