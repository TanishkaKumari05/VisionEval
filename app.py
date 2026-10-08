import streamlit as st
import torch
import pandas as pd
from PIL import Image
from io import BytesIO

from transformers import (
    CLIPModel,
    CLIPProcessor,
    BlipProcessor,
    BlipForConditionalGeneration
)

st.set_page_config(
    page_title="VisionEval",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 VisionEval")
st.subheader("Multimodal AI Model Evaluation Platform")
st.write("Evaluate vision-language models using images, labels and performance metrics.")

LABELS = [
    "dog", "cat", "car", "person",
    "bird", "building", "computer", "padlock"
]

@st.cache_resource
def load_clip():
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor


def predict_clip(image, model, processor):
    prompts = [f"a photo of a {label}" for label in LABELS]

    inputs = processor(
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.inference_mode():
        outputs = model(**inputs)
        probabilities = outputs.logits_per_image.softmax(dim=1)[0]

    best_index = probabilities.argmax().item()

    return (
        LABELS[best_index],
        probabilities[best_index].item() * 100,
        {
            LABELS[i]: probabilities[i].item() * 100
            for i in range(len(LABELS))
        }
    )


st.divider()
task = st.selectbox(
    "Select Evaluation Task",
    [
        "Image Classification",
        "Image Captioning",
        "Visual Question Answering"
    ]
)

if task == "Image Classification":

    mode = st.radio(
        "Evaluation Mode",
        ["Single Image", "Multiple Images"],
        horizontal=True
    )

    if mode == "Single Image":

        st.header("📷 Single Image Evaluation")

        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"],
            key="single_upload"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, width=400)

            ground_truth = st.selectbox(
                "Select Ground Truth",
                LABELS,
                key="single_truth"
            )

            if st.button("Run CLIP Classification 🚀"):

                with st.spinner("Running CLIP..."):
                    model, processor = load_clip()
                    prediction, confidence, scores = predict_clip(
                        image, model, processor
                    )

                st.session_state["single_result"] = {
                    "filename": uploaded_file.name,
                    "prediction": prediction,
                    "ground_truth": ground_truth,
                    "confidence": confidence,
                    "scores": scores
                }

            if "single_result" in st.session_state:
                result = st.session_state["single_result"]

                if result["filename"] == uploaded_file.name:

                    prediction = result["prediction"]
                    truth = result["ground_truth"]
                    confidence = result["confidence"]
                    correct = prediction == truth

                    st.divider()
                    st.header("📊 CLIP Results Dashboard")

                    col1, col2, col3 = st.columns(3)

                    col1.metric("Prediction", prediction.title())
                    col2.metric("Ground Truth", truth.title())
                    col3.metric(
                        "Result",
                        "Correct ✅" if correct else "Wrong ❌"
                    )

                    col4, col5 = st.columns(2)
                    col4.metric("Confidence", f"{confidence:.2f}%")
                    col5.metric(
                        "Single-Image Score",
                        "100%" if correct else "0%"
                    )

                    if correct:
                        st.success("CLIP correctly classified this image.")
                    else:
                        st.warning(
                            f"CLIP predicted {prediction}, "
                            f"but the correct label is {truth}."
                        )

                    st.subheader("Class Probability Distribution")
                    st.bar_chart(
                        pd.DataFrame(
                            list(result["scores"].items()),
                            columns=["Class", "Probability (%)"]
                        ).set_index("Class")
                    )

    else:

        st.header("🖼️ Multi-Image CLIP Evaluation")

        st.write(
            "Upload multiple images and assign the correct "
            "class to each image before running the evaluation."
        )

        uploaded_files = st.file_uploader(
            "Upload images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="batch_upload"
        )

        if uploaded_files:

            if len(uploaded_files) > 20:
                st.warning(
                    "Please upload a maximum of 20 images per batch."
                )
                st.stop()

            st.subheader("Assign Ground Truth Labels")

            ground_truths = []

            for i, file in enumerate(uploaded_files):

                col1, col2 = st.columns([1, 3])

                with col1:
                    st.image(file, width=120)

                with col2:
                    st.write(f"**Image {i+1}: {file.name}**")
                    label = st.selectbox(
                        "Correct Class",
                        LABELS,
                        key=f"truth_{i}_{file.file_id}"
                    )
                    ground_truths.append(label)

            if st.button("🚀 Run Batch Evaluation"):

                results = []

                with st.spinner("Loading CLIP model..."):
                    model, processor = load_clip()

                progress = st.progress(0)

                for i, file in enumerate(uploaded_files):

                    try:
                        image = Image.open(
                            BytesIO(file.getvalue())
                        ).convert("RGB")

                        prediction, confidence, scores = predict_clip(
                            image, model, processor
                        )

                        truth = ground_truths[i]
                        correct = prediction == truth

                        results.append({
                            "Image": file.name,
                            "Ground Truth": truth,
                            "Prediction": prediction,
                            "Confidence (%)": round(confidence, 2),
                            "Correct": correct
                        })

                    except Exception as error:
                        st.error(
                            f"Could not evaluate {file.name}: {error}"
                        )

                    progress.progress((i + 1) / len(uploaded_files))

                st.session_state["batch_results"] = results
                st.session_state["batch_signature"] = [
                    (file.file_id, label)
                    for file, label in zip(uploaded_files, ground_truths)
                ]

            current_signature = [
                (file.file_id, label)
                for file, label in zip(uploaded_files, ground_truths)
            ]

            if (
                "batch_results" in st.session_state
                and st.session_state.get("batch_signature")
                == current_signature
            ):

                df = pd.DataFrame(st.session_state["batch_results"])

                if not df.empty:

                    total = len(df)
                    correct_count = int(df["Correct"].sum())
                    incorrect_count = total - correct_count
                    accuracy = correct_count / total * 100
                    avg_confidence = df["Confidence (%)"].mean()

                    st.divider()
                    st.header("📊 Batch Evaluation Dashboard")

                    c1, c2, c3, c4 = st.columns(4)

                    c1.metric("Total Images", total)
                    c2.metric("Correct", correct_count)
                    c3.metric("Incorrect", incorrect_count)
                    c4.metric("Accuracy", f"{accuracy:.2f}%")

                    st.metric(
                        "Average Confidence",
                        f"{avg_confidence:.2f}%"
                    )

                    st.subheader("Correct vs Incorrect Predictions")

                    chart_data = pd.DataFrame({
                        "Result": ["Correct", "Incorrect"],
                        "Images": [correct_count, incorrect_count]
                    }).set_index("Result")

                    st.bar_chart(chart_data)

                    st.subheader("📋 Detailed Evaluation Results")
                    st.dataframe(df, use_container_width=True)

                    csv = df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        "⬇️ Download Results as CSV",
                        data=csv,
                        file_name="visioneval_clip_results.csv",
                        mime="text/csv"
                    )

                    st.info(
                        "Accuracy is calculated across successfully "
                        "evaluated images only."
                    )

elif task == "Image Captioning":
    st.info("🚧 BLIP image captioning is coming next.")

elif task == "Visual Question Answering":
    st.info("🚧 LLaVA and Gemini evaluation is coming next.")
    # -------------------------------
# BLIP IMAGE CAPTIONING
# -------------------------------

if uploaded_image:

    st.divider()
    st.header("📝 BLIP Image Captioning")

    if st.button("Run BLIP Captioning 🚀"):

        with st.spinner("Loading BLIP model..."):

            blip_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )

            blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )

            image = Image.open(uploaded_image).convert("RGB")

            inputs = blip_processor(
                images=image,
                return_tensors="pt"
            )

            with torch.no_grad():
                output = blip_model.generate(
                    **inputs,
                    max_new_tokens=30
                )

            caption = blip_processor.decode(
                output[0],
                skip_special_tokens=True
            )

        st.success("BLIP caption generated successfully! ✅")

        st.subheader("Generated Caption")
        st.write(caption)

        st.session_state["blip_caption"] = caption
