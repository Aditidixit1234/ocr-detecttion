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
        """Perform OCR on an image and return the recognized text and confidence scores."""
        image = Image.open(image_path).convert("RGB")
        img_np = np.array(image)
        
        # Use EasyOCR to detect text regions
        print("Detecting text regions...")
        # Increase sensitivity by lowering text_threshold
        # and keep paragraph=False to get more granular boxes
        detections = self.reader.readtext(
            img_np, 
            detail=1, 
            text_threshold=0.5,
            link_threshold=0.3,
            low_text=0.3
        )
        
        if not detections:
            print("No text detected by EasyOCR, trying fallback segmentation...")
            text = self._process_fallback(image)
            return text, [0.5] # Default low confidence for fallback

        lines = self._group_detections_to_lines(detections)
        
        results = [] # List of (text, confidence)
        print(f"Processing {len(lines)} lines of text...")
        for line in lines:
            min_x = min([min([p[0] for p in d[0]]) for d in line])
            max_x = max([max([p[0] for p in d[0]]) for d in line])
            min_y = min([min([p[1] for p in d[0]]) for d in line])
            max_y = max([max([p[1] for p in d[0]]) for d in line])
            
            padding = 5
            min_x = max(0, int(min_x - padding))
            min_y = max(0, int(min_y - padding))
            max_x = min(img_np.shape[1], int(max_x + padding))
            max_y = min(img_np.shape[0], int(max_y + padding))
            
            line_crop = image.crop((min_x, min_y, max_x, max_y))
            pixel_values = self.processor(line_crop, return_tensors="pt").pixel_values.to(self.device)
            
            # Generate with scores
            outputs = self.model.generate(
                pixel_values, 
                max_new_tokens=128, 
                num_beams=4, 
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            generated_ids = outputs.sequences
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Calculate confidence from scores
            # scores is a tuple of (num_tokens) length, each element is [batch, vocab_size]
            # For simplicity, we'll take the average of the log-probs of selected tokens
            if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
                # sequences_scores is log-probability of the whole sequence
                # Convert to a 0-1 range roughly (exp of average log-prob)
                seq_len = generated_ids.shape[1]
                conf = torch.exp(outputs.sequences_scores / seq_len).item()
            else:
                conf = 0.9 # Fallback
            
            if generated_text.strip():
                results.append((generated_text.strip(), conf))
        
        full_text = " ".join([r[0] for r in results])
        scores = [r[1] for r in results]
        
        return full_text, scores

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
