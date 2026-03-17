import torch
import numpy as np
import cv2
import easyocr
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class OCRSystem:
    def __init__(self, model_name="microsoft/trocr-base-handwritten"):
        print(f"Loading OCR Model: {model_name}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        
        # Initialize EasyOCR for detection (CPU by default if no GPU)
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

    def _group_detections_to_lines(self, detections, y_threshold=20):
        """Group detected boxes into lines based on their vertical positions."""
        # detections: list of [bbox, text, conf]
        # bbox: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        
        if not detections:
            return []
            
        # Sort detections by Y-coordinate first
        detections.sort(key=lambda x: x[0][0][1])
        
        lines = []
        if not detections:
            return lines
            
        current_line = [detections[0]]
        for i in range(1, len(detections)):
            prev_y = detections[i-1][0][0][1]
            curr_y = detections[i][0][0][1]
            
            if abs(curr_y - prev_y) < y_threshold:
                current_line.append(detections[i])
            else:
                # Sort current line by X-coordinate
                current_line.sort(key=lambda x: x[0][0][0])
                lines.append(current_line)
                current_line = [detections[i]]
        
        if current_line:
            current_line.sort(key=lambda x: x[0][0][0])
            lines.append(current_line)
            
        return lines

    def process_image(self, image_path):
        """Perform OCR on an image and return the recognized text."""
        image = Image.open(image_path).convert("RGB")
        img_np = np.array(image)
        
        # Use EasyOCR to detect text regions
        # detail=1 returns bounding boxes
        print("Detecting text regions...")
        detections = self.reader.readtext(img_np, detail=1)
        
        if not detections:
            print("No text detected by EasyOCR, trying fallback segmentation...")
            # Fallback to simple segmentation if EasyOCR fails
            return self._process_fallback(image)

        # Group detections into lines
        lines = self._group_detections_to_lines(detections)
        
        all_text = []
        print(f"Processing {len(lines)} lines of text...")
        for line in lines:
            # For each line, we could either process word by word or crop the whole line
            # TrOCR is better at lines, so let's crop the line bounding box
            
            # Find the bounding box of the whole line
            min_x = min([min([p[0] for p in d[0]]) for d in line])
            max_x = max([max([p[0] for p in d[0]]) for d in line])
            min_y = min([min([p[1] for p in d[0]]) for d in line])
            max_y = max([max([p[1] for p in d[0]]) for d in line])
            
            # Add some padding
            padding = 5
            min_x = max(0, int(min_x - padding))
            min_y = max(0, int(min_y - padding))
            max_x = min(img_np.shape[1], int(max_x + padding))
            max_y = min(img_np.shape[0], int(max_y + padding))
            
            line_crop = image.crop((min_x, min_y, max_x, max_y))
            
            pixel_values = self.processor(line_crop, return_tensors="pt").pixel_values.to(self.device)
            
            generated_ids = self.model.generate(
                pixel_values, 
                max_new_tokens=128, 
                num_beams=4, 
                early_stopping=True
            )
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            if generated_text.strip():
                all_text.append(generated_text.strip())
        
        return " ".join(all_text)

    def _process_fallback(self, image):
        """Fallback processing for images where EasyOCR detection fails."""
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        generated_ids = self.model.generate(
            pixel_values, 
            max_new_tokens=128, 
            num_beams=4, 
            early_stopping=True
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.strip()
