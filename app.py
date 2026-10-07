import streamlit as st
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

st.set_page_config(
    page_title="VisionEval",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 VisionEval")
st.subheader("Multimodal AI Model Evaluation Platform")

st.write(
    "Compare multimodal AI models and analyze their performance "
    "across different vision-language tasks."
)

st.divider()

# -------------------------------
# Upload Image
# -------------------------------

st.header("Upload an Image")

uploaded_image = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image:
    st.image(
        uploaded_image,
        caption="Uploaded Image",
        width=500
    )

    st.success("Image uploaded successfully! ✅")

# -------------------------------
# Select Task
# -------------------------------

st.divider()

st.header("Select Evaluation Task")

task = st.selectbox(
    "Choose a task",
    [
        "Image Classification",
        "Image Captioning",
        "Visual Question Answering"
    ]
)

st.write("Selected task:", task)

# -------------------------------
# CLIP Image Classification
# -------------------------------

if uploaded_image and task == "Image Classification":

    st.divider()
    st.header("🤖 CLIP Evaluation")

    if st.button("Run CLIP Classification 🚀"):

        with st.spinner("Loading CLIP model..."):

            model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            )

            processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )

        image = Image.open(uploaded_image).convert("RGB")

        labels = [
            "a photo of a dog",
            "a photo of a cat",
            "a photo of a car",
            "a photo of a person",
            "a photo of a bird",
            "a photo of a building",
            "a photo of a computer",
            "a photo of a padlock"
        ]

        inputs = processor(
            text=labels,
            images=image,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():

            outputs = model(**inputs)

            logits_per_image = outputs.logits_per_image

            probabilities = logits_per_image.softmax(dim=1)[0]

        best_index = probabilities.argmax().item()

        prediction = labels[best_index]

        confidence = probabilities[best_index].item() * 100

        st.success("CLIP evaluation completed! ✅")

        st.subheader("Prediction")

        st.write(
            f"**Predicted Class:** {prediction}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.subheader("Class Probabilities")

        results = {
            labels[i]: f"{probabilities[i].item() * 100:.2f}%"
            for i in range(len(labels))
        }

        st.json(results)# -------------------------------
# Ground Truth Evaluation
# -------------------------------

st.divider()

st.subheader("🎯 Ground Truth Evaluation")

ground_truth = st.selectbox(
    "Select the correct class (Ground Truth):",
    labels
)

if st.button("Evaluate Prediction 📊"):

    predicted_class = prediction.replace("a photo of a ", "").strip()
    true_class = ground_truth.replace("a photo of a ", "").strip()

    if predicted_class == true_class:

        st.success("✅ Correct Prediction!")

        st.metric(
            "Accuracy",
            "100%"
        )

    else:

        st.error("❌ Incorrect Prediction")

        st.metric(
            "Accuracy",
            "0%"
        )

        st.write(
            f"**Ground Truth:** {true_class}"
        )

        st.write(
            f"**CLIP Prediction:** {predicted_class}"
        )
