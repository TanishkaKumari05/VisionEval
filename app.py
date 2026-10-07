import streamlit as st

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
