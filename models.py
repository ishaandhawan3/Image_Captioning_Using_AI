import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Set Hugging Face token
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption
