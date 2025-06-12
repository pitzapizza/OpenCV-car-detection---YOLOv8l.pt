# OpenCV Car Detection using YOLOv8l

This project implements a real-time car detection system using OpenCV and YOLOv8l, focused on identifying only cars from static images with high accuracy. It combines traditional image processing techniques with deep learning-based object detection for a smart and scalable vision-based solution.

## 🚀 Project Objective

The aim of this project is to detect cars in input images using the YOLOv8l object detection model. The system preprocesses the input image, filters out all non-car objects, and highlights detected cars using bounding boxes with adjustable label scaling. This approach is useful for traffic analysis, vehicle tracking, and smart surveillance systems.

## 🧠 Technologies Used

- OpenCV: For image manipulation, preprocessing, color conversion, bounding box rendering, and display.
- Ultralytics YOLOv8l: A high-accuracy deep learning model used for object detection.
- Python: Primary language used for scripting and integration.

## 📁 Project Structure

```
project-root/
├── src/
│   ├── images/                        # Input test images
│   ├── detected_cars_images/         # Output folder for saved detection results
│   └── models/
│       └── yolov8l.pt                # YOLOv8l model (excluded from GitHub)
├── main_script.py                    # Car detection logic
└── README.md                         # Project documentation
```

## 🛠 How It Works

1. **Image Loading and Preprocessing**  
   - The image is read using `cv2.imread()`  
   - If too large, it’s resized to fit within screen dimensions  
   - Converted from BGR to RGB  
   - Pixel values normalized to range [0, 1]  

2. **YOLOv8 Car Detection**  
   - The model detects all objects, but only class `2` (car) is processed  
   - Bounding boxes are drawn with confidence labels  
   - Labels adjust in size relative to object width for clarity  

3. **Output Handling**  
   - Processed images are saved to the `detected_cars_images` folder  
   - Preview window displays the detection results  

## 🔗 Dependencies and Installation

Install the required Python packages before running:

```bash
pip install opencv-python numpy ultralytics
```

Download the YOLOv8l model from the official Ultralytics release page:  
https://github.com/ultralytics/ultralytics  
Save the file as `yolov8l.pt` inside `src/models/`

## 📈 Results and Observations

- Successfully detects rear, front, and side views of cars with good precision
- Ignores non-car objects like motorcycles, buses, and trucks
- Confidence thresholding reduces false positives
- Dynamic label sizing improves visualization clarity
- Code structure is modular, readable, and easy to extend

## 🙋‍♂️ About the Author

This project was developed by an engineering student exploring computer vision and deep learning for smart traffic systems. It serves as a stepping stone for more advanced solutions like real-time traffic monitoring or AI-driven parking management tools.

## 🧪 Future Improvements

- Batch detection support for multiple images  
- Real-time detection using webcam or video feeds  
- Fine-tuning the model on custom datasets for region-specific accuracy  
- Integration with Flask for web-based demos
