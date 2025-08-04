import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr
from io import BytesIO
from PIL import Image

# --- Load decoder model ---
decoder = tf.keras.models.load_model("vae_decoder.h5", compile=False)
latent_dim = decoder.input_shape[1]

# --- Function to generate and return image as PIL ---
def generate_plot(n_images=10):
    # Sample random latent vectors
    random_latent_vectors = np.random.normal(size=(n_images, latent_dim))
    generated_images = decoder.predict(random_latent_vectors)

    # Create matplotlib figure
    fig, axes = plt.subplots(1, n_images, figsize=(n_images * 2, 2))
    if n_images == 1:
        axes = [axes] 

    for i in range(n_images):
        image = generated_images[i]
        if image.shape[-1] == 1:  # if grayscale image with shape (H, W, 1)
            image = image.squeeze()
            axes[i].imshow(image, cmap="gray")
        else:
            axes[i].imshow(image)
        axes[i].axis("off")

    plt.tight_layout()

    # Save figure to a BytesIO buffer
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    # Return PIL Image for Gradio
    return Image.open(buf)

# --- Gradio Interface ---
demo = gr.Interface(
    fn=generate_plot,
    inputs=gr.Slider(1, 10, step=1, value=5, label="Number of Images"),
    outputs=gr.Image(type="pil", label="Generated Images"),
    title="VAE Image Generator",
    description="Generates a row of images from the VAE decoder."
)

if __name__ == "__main__":
    demo.launch()
