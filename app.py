import gradio as gr
from models import model

def generate_caption(image):
    return model.generate_caption(image)

# Create Gradio interface
demo = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Image Captioning with Hugging Face Model",
    description="Upload an image to generate a caption",
)
