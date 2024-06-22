# IOT-YOLO-Project
## Team10 : Boat Detection & Prediction System

### Archive

1. YOUTUBE LINK

2. PPT SLIDES DOWNLOAD

3. DATASETS

- DOWNLOAD :


- ROBOFLOW :
   - AERIAL 1&2 : https://universe.roboflow.com/public-1tecx/boat-detection-bzj6m/dataset/1 
   - TRACKING 1 : https://universe.roboflow.com/boats-hu0jt/boat-tracking-model/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true
   - TRACKING 2 : https://universe.roboflow.com/shipdataset3-3prli/shipdata-a4ucj/dataset/9

5. VIDEOS
DOWNLOAD : 


### PROJECT WORKFLOW

<img width="696" alt="스크린샷 2024-06-22 오후 5 55 22" src="https://github.com/jhjh-kim/IOT-YOLO-Project/assets/132179356/d7307214-c0fe-487e-9aa6-6bc0a29e80b6">

## PROJECT STEPS
1. PROJECT TOPIC

2. DATASETS

3. SELECTING YOLOv8 MODEL

4. COMPARING MODEL PERORMANCE BY LOSS FUNCTION
   
5. USING TRANSFER LEARNING

6. MODEL EVALUTATION

7. APPLICATION


### 1. PROJECT TOPIC
#### Background
Illegal fishing boats frequently appear along the Northern Limit Line (NLL), the maritime boundary between South Korea and North Korea. Given the issue of decreasing military personnel in South Korea, we have devised a solution to address this challenge: an unmanned drone system utilizing computer vision technology.

#### Goal & Output
Implement the idea of using the YOLOv8 model to detect boats in the ocean and change their status when they move towards or cross significant points like the NLL, utilizing Deep SORT, and provide this information in real-time.



### 2. Datasets
- DATASET SELECTION : The dataset is composed of a total of three datasets: two publicly available datasets from ROBOFLOW and one dataset created by capturing images ourselves. When selecting the datasets, we considered various types of boats, sizes, and angles.

- DATA PROCESSING : As will be mentioned later, to improve the model's performance, we added images that capture the phenomena when boats are in motion. Additionally, we added noise to the images and manually labeled and filtered out low-quality datasets.
<img width="1012" alt="스크린샷 2024-06-22 오후 8 27 42" src="https://github.com/jhjh-kim/IOT-YOLO-Project/assets/132179356/738647db-17be-497b-9bf9-c8c5c8b147a0">



### 3. Selecting YOLOv8 Model
<img width="715" alt="스크린샷 2024-06-22 오후 1 07 20" src="https://github.com/jhjh-kim/IOT-YOLO-Project/assets/132179356/9ac6dec0-b27b-44de-b099-10e3d49c7dd0">

To implement a real-time system, we selected a small model that satisfies both high accuracy and high speed.

Additionally, when training on the demo dataset with four different loss functions, we found that the BCE loss function performed the best. Therefore, we decided to use the BCE loss function for future training.



### 4. Comparing Model Performance by Loss Function
When comparing the model performance with four different loss functions on the same demo dataset, we found that the BCE loss function performed the best. Therefore, we decided to use the BCE loss function for future training.



### 5. Using Transfer Learning
To reflect various types, sizes, and angles of boats, we conducted a total of four transfer learning stages using different datasets and parameters.


<img width="698" alt="스크린샷 2024-06-22 오후 7 02 24" src="https://github.com/jhjh-kim/IOT-YOLO-Project/assets/132179356/94409487-fcdc-4b6f-979c-330513c48743">



### 6. Final Model Performance


### 7. Application
The drone operator sets the position to move the drone and selects the mode based on either an aerial or side view.
 - video1 : aerial view + dynamic movements
 - video2 : side view + dynamic movements
 - video3 : aerial view + multiple boats


Depending on the location and shooting angle, there are significant boundary lines. If a boat crosses these significant boundaries or its direction of travel points towards the boundary, the color of its bounding box changes.
- Green bbox: Normal state
- Yellow bbox: Caution state
- Red bbox: Danger state
- Blue line: Predicted movement direction of the boat


## USING SCIPRT FILES
0. prepare your dataset and .yaml file.

1. cd ~/yolov8

2. pip install -e .

3. go to ~/yolov8/ultralytics/utils/loss.py

4. go to class v8DetectionLoss
   - in def __init__, I initialized avaialble loss functions.
   - adjust the class number parameter of OneStageSeesawLoss according to your dataset.
   - in def __call__, uncomment the codes of a loss function and test the model performance (You don't need to adjust anything here).

5. go to ~/yolov8/ultralytics/models/yolo/custom_train.py
   - adjust the training setting as you want.
   - execute the script file by "python custom_train.py"
  
## SCRIPT FILES EXPLANATION



## REFRENCES
1. DATASETS
- AERIAL 1&2 : https://universe.roboflow.com/public-1tecx/boat-detection-bzj6m/dataset/1 
- TRACKING 1 : https://universe.roboflow.com/boats-hu0jt/boat-tracking-model/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true
- TRACKING 2 : https://universe.roboflow.com/shipdataset3-3prli/shipdata-a4ucj/dataset/9

  
2. VIDEOS
- Aerial 1 :
- Side 1 : 
- Aerial 2 : https://youtu.be/u6uNECa_X8Q?si=11l7nUsRdM9geGmJ

3. CODES
https://docs.ultralytics.com/
