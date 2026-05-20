import streamlit as st
from PIL import Image
from src.similarity import find_similar_images

st.set_page_config(page_title="StyleMatch AI", page_icon="👕", layout="centered")


st.title("👕 StyleMatch AI")
st.write("Upload a clothing image and find similar products!")

uploaded_file = st.file_uploader("Choose an clothing image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(uploaded_image, caption="Your uploaded image", use_container_width=True)

    with st.spinner("Finding similar clothing items..."):
        similar_images = find_similar_images(uploaded_image, top_k=2)

    st.subheader("Top 2 Similar Items")

    for path, score in similar_images:
        product_image = Image.open(path)

        st.image(
            product_image,
            caption=f"{path.stem} (Similarity: {score:.4f})",
            use_container_width=True
        )