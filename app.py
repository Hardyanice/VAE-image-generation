import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load decoder only
@st.cache_resource
def load_decoder():
    return tf.keras.models.load_model("vae_decoder.h5", compile=False)

decoder = load_decoder()
latent_dim = decoder.input_shape[1]

st.title("VAE Image Generator")
n_images = st.slider("Number of images to generate", 1, 10, 5)

if st.button("Generate"):
    random_latent_vectors = np.random.normal(size=(n_images, latent_dim))
    generated_images = decoder.predict(random_latent_vectors)

    fig, axes = plt.subplots(1, n_images, figsize=(n_images * 2, 2))
    for i in range(n_images):
        axes[i].imshow(generated_images[i].squeeze(), cmap="gray")
        axes[i].axis("off")
    st.pyplot(fig)
