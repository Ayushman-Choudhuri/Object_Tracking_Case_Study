import cv2
import numpy as np
import yaml 

class YOLOv3Detector(): 

    def __init__(self, model_config_path, model_weights_path):

        #Load YOLOv3 Model from cv2.dnn

        self.net = cv2.dnn.readNetFromDarknet(model_config_path, model_weights_path)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[layer[0] - 1] for layer in self.net.getUnconnectedOutLayers()]

    def _frame_preprocess(self, frame): #private

        pass


    def detect_potato(self, frame): 
        
        pass 


if __name__ == "__main__":
    
    pass