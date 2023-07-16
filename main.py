import cv2 
import os
import sys
import yaml

from src.detection import YOLOv3PotatoDetector
from src.tracker import PotatoTracker

def main(): 
    #Initialize paths

    # Add the src directory to the module search path
    sys.path.append(os.path.abspath('src'))
    sys.path.append(os.path.abspath('config'))

    #Initialize Detector 
    model_config_path = 'config/yolov3_testing.cfg'
    model_weights_path = 'weights/yolov3_training.weights'
    detector = YOLOv3PotatoDetector(model_config_path, model_weights_path)
    tracker = PotatoTracker()
    print("Models Initialized")

    #Open the video file
    video_path = 'data/video.mp4'
    video = cv2.VideoCapture(video_path)

    while True:
        # Read a frame from the video
        ret, frame = video.read()

        if not ret:
            break

        # Perform object detection on the extracted frame from video
        detected_frame = detector.detect_potato(frame)

        # Get the detections from the potato detector
        detections = detector.get_detections()

        # Display the frame with detected objects
        cv2.imshow('Object Detection and Tracking', frame)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()