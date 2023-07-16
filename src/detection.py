import cv2
import numpy as np
#import yaml 

class YOLOv3PotatoDetector(): 
    def __init__(self, model_config_path, model_weights_path):
        # Load YOLOv3 Model from cv2.dnn
        self.net = cv2.dnn.readNetFromDarknet(model_config_path, model_weights_path)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = outputlayers = [self.layer_names[i-1] for i in self.net.getUnconnectedOutLayers()]
        self.detections = []

    def _frame_to_blob(self, frame): #private
        #Convert frame to blob
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        return blob
    
    def _apply_bbox_nms(): 
        pass #to be imlpemented


    def detect_potato(self, frame): 

        #Clear detections list
        self.detections = []
        
        #Convert frame to blob
        blob_input = self._frame_to_blob(frame)

        # Set the input to the network
        self.net.setInput(blob_input)

        # Run forward pass through the yolov3 model
        yolo_outputs = self.net.forward(self.output_layers)
        
        for outputs in yolo_outputs: 
            for detection in outputs: 
                scores = detection[5:]
                class_id = scores.argmax()
                confidence = scores[class_id]

                #Filter through a confidence threshold
                if confidence > 0.5: 
                    #print(detection)
                    
                    # Get the bounding box coordinates
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    width = int(detection[2] * frame.shape[1])
                    height = int(detection[3] * frame.shape[0])

                    # Calculate the top-left corner of the bounding box
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    # Draw the bounding box and label on the frame
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    cv2.putText(frame, str("potato"), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                     # Append the bounding box coordinates to detections
                    self.detections.append(([x, y, width, height], confidence, "potato")) 
        #print(self.detections)
        #print(".............................")       
        return frame 
    
    def get_detections(self):
        
        return self.detections


if __name__ == "__main__":
    
    pass