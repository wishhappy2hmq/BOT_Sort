YOLO + BOTSORT Object Tracking
This project demonstrates how to perform object tracking using the YOLO (You Only Look Once) object detection model and the BOTSORT tracking algorithm. The YOLO model is used to detect objects in each frame of a video, and BOTSORT is used for tracking those objects across frames.

Features
YOLO Detection: Detect objects in each video frame.
BOTSORT Tracking: Track objects across frames using the BOTSORT algorithm.
Real-time Display: Display the detected and tracked objects in real-time.
Track Information: Show unique track IDs and bounding boxes for each tracked object.
Requirements
Python 3.7+
OpenCV
ultralytics (for YOLO)
numpy
BOTSORT (for tracking)
You can install the required dependencies using the following commands:

pip install opencv-python ultralytics numpy
To install BOTSORT, you'll need to clone the repository or install it manually (depending on its distribution method).

Usage
Command Line Arguments
--video : Path to the input video file.
--model : Path to the YOLO model file (e.g., yolov8.pt).
Example Command
python track_objects.py --video path_to_video.mp4 --model path_to_yolo_model.pt
This will load the specified YOLO model and the video file, then process the video with object detection and tracking.

File Structure
track_objects.py: The main script for running object tracking.
utils.py: Contains helper functions (if any).
requirements.txt: A file listing the required Python packages.
How It Works
YOLO Model: The YOLO model is used to detect objects in each video frame. The detection results include bounding boxes with coordinates (x_min, y_min, x_max, y_max) and confidence scores.
BOTSORT Tracker: The BOTSORT tracker receives the detection results and tracks the objects across frames. Each object gets a unique track ID.
Drawing Bounding Boxes: For each tracked object, a bounding box is drawn on the video frame with the corresponding track ID displayed above it.
Real-time Display: The processed frames are displayed in real-time using OpenCV's imshow function.
Notes
Performance: The tracking algorithm may require a high-performance machine for real-time video processing, especially if using large or high-resolution videos.
Model File: Ensure that the YOLO model file is correctly trained and compatible with your use case. You can use pre-trained YOLOv8 models or fine-tune a model for your specific needs.
License
This project is licensed under the MIT License - see the LICENSE file for details.
