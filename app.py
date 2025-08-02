import gradio as gr
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load only decoder model
model = tf.keras.models.load_model("vae_model.h5")

def generate_digit(z1, z2):
    z = np.array([[z1, z2]])
    image = model.predict(z)[0]
    return image.squeeze()

# Interface
iface = gr.Interface(
    fn=generate_digit,
    inputs=[
        gr.Slider(-3, 3, step=0.1, label="Latent z[0]"),
        gr.Slider(-3, 3, step=0.1, label="Latent z[1]")
    ],
    outputs=gr.Image(shape=(28, 28), image_mode='L', label="Generated Digit"),
    title="VAE Digit Generator",
    description="Move the sliders to explore the VAE latent space."
)

if __name__ == "__main__":
    iface.launch()
