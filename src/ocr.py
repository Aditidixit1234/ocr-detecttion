import torch
import numpy as np
import easyocr
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class OCRSystem:
    def __init__(self, model_name="microsoft/trocr-base-handwritten"):
        print(f"Loading OCR Model: {model_name}...")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)

        # 🔥 Enable FP16 if GPU
        if self.device.type == "cuda":
            self.model = self.model.half()

        self.model.eval()

        # EasyOCR (GPU if available)
        self.reader = easyocr.Reader(['en'], gpu=self.device.type == "cuda")

    def _group_detections_to_lines(self, detections, y_threshold=20):
        if not detections:
            return []

        detections.sort(key=lambda x: x[0][0][1])

        lines = []
        current_line = [detections[0]]

        for i in range(1, len(detections)):
            prev_y = detections[i-1][0][0][1]
            curr_y = detections[i][0][0][1]

            if abs(curr_y - prev_y) < y_threshold:
                current_line.append(detections[i])
            else:
                current_line.sort(key=lambda x: x[0][0][0])
                lines.append(current_line)
                current_line = [detections[i]]

        if current_line:
            current_line.sort(key=lambda x: x[0][0][0])
            lines.append(current_line)

        return lines

    def process_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        img_np = np.array(image)

        print("Detecting text regions...")
        detections = self.reader.readtext(img_np, detail=1)

        if not detections:
            print("Fallback triggered...")
            return self._process_fallback(image), [0.5]

        lines = self._group_detections_to_lines(detections)

        # 🔥 Collect ALL line crops first (batching)
        crops = []
        for line in lines:
            min_x = min([min([p[0] for p in d[0]]) for d in line])
            max_x = max([max([p[0] for p in d[0]]) for d in line])
            min_y = min([min([p[1] for p in d[0]]) for d in line])
            max_y = max([max([p[1] for p in d[0]]) for d in line])

            padding = 5
            crop = image.crop((
                max(0, int(min_x - padding)),
                max(0, int(min_y - padding)),
                min(img_np.shape[1], int(max_x + padding)),
                min(img_np.shape[0], int(max_y + padding))
            ))

            crops.append(crop)

        # 🔥 Batch processing
        with torch.no_grad():
            pixel_values = self.processor(crops, return_tensors="pt", padding=True).pixel_values.to(self.device)

            if self.device.type == "cuda":
                pixel_values = pixel_values.half()

            outputs = self.model.generate(
                pixel_values,
                max_new_tokens=128,
                num_beams=4,
                early_stopping=True
            )

        texts = self.processor.batch_decode(outputs, skip_special_tokens=True)

        results = [(t.strip(), 0.9) for t in texts if t.strip()]

        full_text = " ".join([r[0] for r in results])
        scores = [r[1] for r in results]

        if len(full_text.split()) < 5:
            print("Fallback triggered...")
            fallback_text = self._process_fallback(image)
            if len(fallback_text.split()) > len(full_text.split()):
                return fallback_text, [0.7]

        return full_text, scores

    def _process_fallback(self, image):
        with torch.no_grad():
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)

            if self.device.type == "cuda":
                pixel_values = pixel_values.half()

            outputs = self.model.generate(pixel_values, max_new_tokens=128)

        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()