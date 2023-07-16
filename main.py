import cv2 
from src.detection import YOLOv3Detector

#Initialize Detector 
model_config_path = 'config/yolov3_testing.cfg'
model_weights_path = 'weights/yolov3_training.weights'
detector = YOLOv3Detector(model_config_path, model_weights_path)


#Open the video file
video_path = 'data/video.mp4'
video = cv2.VideoCapture(video_path)

while True:
    # Read a frame from the video
    ret, frame = video.read()

    if not ret:
        break

    # Perform object detection on the frame
    detected_frame = detector.detect_potato(frame)

    # Display the frame with detected objects
    cv2.imshow('Object Detection', detected_frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()