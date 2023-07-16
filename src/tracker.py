import cv2
import numpy as np
#import yaml 

class PotatoTracker: 

    def __init__(self): 
        self.tracker = cv2.TrackerKCF_create()
        self.tracked_objects = []

    def start_tracker(self, frame, detections): 
        for detection in detections:
            x, y, w, h = detection
            self.tracker.add(frame, (x, y, w, h))

    def update_tracker(self, frame): 
        success, boxes = self.tracker.update(frame)
        self.tracked_objects = []

        for box in boxes:
            x, y, w, h = [int(coord) for coord in box]
            self.tracked_objects.append(((x + w // 2, y + h // 2)))
        print("Tracking status:", success)
        return self.tracked_objects
    
    def draw_track_lines(self, frame):
        for i in range(1, len(self.tracked_objects)):
            prev_center = self.tracked_objects[i - 1]
            curr_center = self.tracked_objects[i]
            cv2.line(frame, prev_center, curr_center, (0, 0, 255), 2)
            cv2.circle(frame, curr_center, 2, (0, 0, 255), -1)


if __name__ == "__main__":
    
    pass