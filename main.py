import cv2 
import os
import sys
import yaml

from src.detection import YOLOv3PotatoDetector
from src.deepsort_tracker import DeepSortTracker


def main(): 
    #Initialize paths
    # Add the src directory to the module search path
    sys.path.append(os.path.abspath('src'))
    sys.path.append(os.path.abspath('config'))

    #Initialize Detector 
    model_config_path = 'config/yolov3_testing.cfg'
    model_weights_path = 'weights/yolov3_training.weights'
    detector = YOLOv3PotatoDetector(model_config_path, model_weights_path)
    potato_tracker = DeepSortTracker()

    # Define a empty dictionary to store the previous center locations for each track ID
    track_history = {} 
    
    print("Models Initialized")

    #Open the video file
    video_path = 'data/video.mp4'
    video = cv2.VideoCapture(video_path)

    while True:
        # Read a frame from the video
        ret, frame_img = video.read()

        if not ret:
            break

        # Perform object detection on the extracted frame from video
        detected_frame = detector.detect_potato(frame_img)

        # Get the detections from the potato detector
        detections = detector.get_detections()
        print(detections)
        print("........................")

        # Object Tracking
        tracks_current = potato_tracker.object_tracker.update_tracks(detections, frame = frame_img)
        potato_tracker.display_track(track_history, tracks_current, frame_img)


        # Display the frame with detected objects
        cv2.imshow('Object Detection and Tracking', frame_img)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()