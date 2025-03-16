from transformers import AutoFeatureExtractor, AutoModelForImageCaptioning
from PIL import Image
import torch

class ImageCaptioningModel:
    def __init__(self, model_name):
        self.model = AutoModelForImageCaptioning.from_pretrained(model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    def generate_caption(self, image):
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        output = self.model.generate(**inputs)
        caption = self.model.config.id2label[output[0].item()]
        return caption

# Initialize the model
model = ImageCaptioningModel("nlpconnect/vit-gpt2-image-captioning")
