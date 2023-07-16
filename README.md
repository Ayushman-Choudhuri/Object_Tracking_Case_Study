# Object_Tracking_Case_Study

### Progress Notes: 
1. The **main** branch contains the updates till 21:00 CET , further updates have been added to the **additions** branch

### Directory Structure
```bash
Object_Tracking_Case_Study
.
├── config
│   ├── config.yml
│   └── yolov3_testing.cfg
├── data
│   ├── dataset
│   └── video.mp4
├── images
├── main.py
├── environment.txt
├── README.md
├── src
│   ├── deepsort_tracker.py
│   ├── detection.py
│   └── tracker.py
└── weights
    └── yolov3_training.weights

```
- Keep the data (videos and images) inside the data folder after creating it.
- Create a weights folder and store the weights inside it. 
- The modules for tracking and detection are kept inside the src folder.

### Dependencies

Run the following command to start a new environment with the required dependencies

```bash
conda create --name potato_tracking --file environment.txt
```

### Running the Project

Execute the following command: 

```bash
python3 main.py
```

### Known Issues: 
1. Currently there are sometimes redundant bounding boxes being detected. The NMS function needs to be tuned in order to avoid redundant bounding boxes
2. When the potato goes under the gripper mechanism in the video, the tracking seems to be lost. The deepSORT tracker parameters need to be tuned for this (most probably the parameters related to Kalman filtering).

