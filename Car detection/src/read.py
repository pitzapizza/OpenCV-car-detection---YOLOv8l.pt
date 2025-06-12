import cv2
import numpy as np
import sys
from ultralytics import YOLO

class ImagePreprocessor:
    def __init__(self, screen_width=1920, screen_height=1080, scaling_ratio=1.5, model_path="src/models/yolov8l.pt"):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.scaling_ratio = scaling_ratio
        self.model = YOLO(model_path)  # Load YOLOv8 model

    def resize_if_necessary(self, image):
        height, width = image.shape[:2]
        resize_required = width > self.screen_width or height > self.screen_height
        
        if resize_required:
            resized = cv2.resize(image, (self.screen_width, self.screen_height), interpolation=cv2.INTER_CUBIC)
            target_width = int(self.screen_width / self.scaling_ratio)
            target_height = int(self.screen_height / self.scaling_ratio)
            scaled = cv2.resize(resized, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
            return scaled
        
        return image

    def preprocess(self, image_path):
        image = cv2.imread(image_path)
        image_missing = image is None

        if image_missing:
            print(f"Error: Could not load image from {image_path}. Check if the file exists.")
            sys.exit(1)

        resized_image = self.resize_if_necessary(image)
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        normalized = image_rgb.astype(np.float32) / 255.0

        return normalized, image_rgb

    def detect_cars(self, image_rgb, conf_threshold=0.25, save_output=True):
        results = self.model.predict(image_rgb, verbose=False)  
        boxes = results[0].boxes.data.cpu().numpy()  

        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            valid_detection = conf > conf_threshold
            
            if cls == 2:  # Only detect cars, ignore other vehicles
                label = "Car"
            else:
                continue

            if valid_detection:
                box_width = x2 - x1
                text_size = max(0.5, box_width / 300)

                cv2.rectangle(image_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(image_rgb, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), 2)

        if save_output:
            output_path = r"D:\OpenCV object detection\Car detection\src\detected_cars_images\detected_car8.jpg"
            cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            print(f"Saved detection output as '{output_path}'.")

        return image_rgb

    def preview_and_exit(self, image):
        cv2.imshow("Processed Image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)  
        cv2.destroyAllWindows()
        sys.exit(0)  

# Usage Example
image_path = r"D:\OpenCV object detection\Car detection\src\images\car8.jpg"  
preprocessor = ImagePreprocessor()
normalized_img, rgb_img = preprocessor.preprocess(image_path)

if normalized_img is not None:
    detected_img = preprocessor.detect_cars(rgb_img)  
    preprocessor.preview_and_exit(detected_img)