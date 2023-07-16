import cv2
import numpy as np
import yaml 

class Tracker: 

    def __init__(self): 
        self.tracker = cv2.TrackerKCF_create()

    def initialize_tracker(self, frame, bbox): 
        pass

    def update(self, frame): 
        pass 


if __name__ == "__main__":
    
    pass