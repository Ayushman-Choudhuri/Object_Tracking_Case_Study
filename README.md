# Object_Tracking_Case_Study

### Progress Notes: 
1. The **main** branch contains the updates till 21:00 CET , further updates and documentation have been added to the **additions** branch

### Known Issues

1. The tracking module could not be integrated in time. Later the tracking implementation was done using the deepSORT tracker instead of the KCF tracker or the other trackers in the cv2 library. This can be found in the additions branch. However the idea of the deepSORT tracking can be found in the presentation. 
2. NMS could not be implemented in time and hence there are redundant bounding boxes. This addition has been made in the additions branch.
3. There are few hard coded figures in the tracker and detector classes, these will be imported from a config.yml file later. 
