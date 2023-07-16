from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import sys
import os
import numpy as np
import yaml
import torch

DISP_OBJ_TRACK_BOX = True
DISP_TRACKS = True

class DeepSortTracker(): 

    def __init__(self):
        
        self.algo_name ="DeepSORT"
        self.object_tracker = DeepSort(max_age=5,
                n_init=1,
                nms_max_overlap=1.0,
                max_cosine_distance=0.3,
                nn_budget=None,
                override_track_class=None,
                embedder='mobilenet',
                half=True,
                bgr=False,
                embedder_gpu=True,
                embedder_model_name=None,
                embedder_wts=None,
                polygon=False,
                today=None)
    
        
    def display_track(self , track_history , tracks_current , img):

        for track in tracks_current:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            
            # Retrieve the current track location(i.e - center of the bounding box) and bounding box
            location = track.to_tlbr()
            bbox = location[:4].astype(int)
            bbox_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

            # Retrieve the previous center location, if available
            prev_centers = track_history.get(track_id ,[])
            prev_centers.append(bbox_center)
            track_history[track_id] = prev_centers
            
            # Draw the track line, if there is a previous center location
            if prev_centers is not None and DISP_TRACKS == True:
                points = np.array(prev_centers, np.int32)
                cv2.polylines(img, [points], False, (51 ,225, 255), 2)

            if DISP_OBJ_TRACK_BOX == True: 
                cv2.rectangle(img,(int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),(0,0,255),1)
                cv2.putText(img, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)