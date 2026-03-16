import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class OCRSystem:
    def __init__(self, model_name="microsoft/trocr-base-handwritten"):
        print(f"Loading OCR Model: {model_name}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)

    def process_image(self, image_path):
        """Perform OCR on an image and return the recognized text."""
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.strip()
