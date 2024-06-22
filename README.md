# IOT-YOLO-Project
## Team10 : Boat Detection & Prediction System

### Archive

1. YOUTUBE LINK :: ==== 첨부필요 ====

2. PPT SLIDES DOWNLOAD :: ==== 첨부필요 ====

3. DATASETS
- DOWNLOAD :: https://drive.google.com/file/d/1s3YDJKxYaUa-Hg_Ek_4feq5ohTH9w_5W/view


- ROBOFLOW ::
   - Aerial 1&2 : https://universe.roboflow.com/public-1tecx/boat-detection-bzj6m/dataset/1 
   - Boat 1 : https://universe.roboflow.com/boats-hu0jt/boat-tracking-model/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true
   - Boat 2 : https://universe.roboflow.com/shipdataset3-3prli/shipdata-a4ucj/dataset/9

5. VIDEOS
DOWNLOAD :: https://drive.google.com/drive/folders/1ETkO5a3iNDmgbmBCXtDy2OGK1IWTmSWP


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
![l_2023051001000357700031231](https://github.com/jhjh-kim/IOT-YOLO-Project/assets/132179356/e7350e04-672d-438c-9567-25b5659501e5)
#### Background
Illegal fishing boats frequently appear along the Northern Limit Line (NLL), the maritime boundary between South Korea and North Korea. Given the issue of decreasing military personnel in South Korea, we have devised a solution to address this challenge: an unmanned drone system utilizing computer vision technology.

#### Goal & Output
Implement the idea of using the YOLOv8 model to detect boats in the ocean and change their status when they move towards or cross significant points like the NLL, utilizing Deep SORT, and provide this information in real-time.



### 2. Datasets
- DATASET SELECTION : The dataset is composed of a total of three datasets: two publicly available datasets from ROBOFLOW and one dataset created by capturing images ourselves. When selecting the datasets, we considered various types of boats, sizes, and angles.

- DATA PROCESSING : To improve the model's performance, we added images that capture the phenomena when boats are in motion. Additionally, we added noise to the images and manually labeled and filtered out low-quality datasets.
<img width="1033" alt="스크린샷 2024-06-22 오후 8 31 31" src="https://github.com/jhjh-kim/IOT-YOLO-Project/assets/132179356/58ea858a-55fd-4abf-bb3a-9c8a5a603631">




### 3. Selecting YOLOv8 Model
<img width="715" alt="스크린샷 2024-06-22 오후 1 07 20" src="https://github.com/jhjh-kim/IOT-YOLO-Project/assets/132179356/9ac6dec0-b27b-44de-b099-10e3d49c7dd0">

To implement a real-time system, we selected a small model that satisfies both high accuracy and high speed.

Additionally, when training on the demo dataset with four different loss functions, we found that the BCE loss function performed the best. Therefore, we decided to use the BCE loss function for future training.




### 4. Comparing Model Performance by Loss Function
<img width="637" alt="스크린샷 2024-06-22 오후 9 16 56" src="https://github.com/jhjh-kim/IOT-YOLO-Project/assets/132179356/fbaa44ab-5c2a-497b-83d0-e4c8f26c5dc6">

When comparing the model performance with four different loss functions on the same demo dataset, we found that the BCE loss function performed the best. Therefore, we decided to use the BCE loss function for future training.



### 5. Using Transfer Learning
<img width="706" alt="model develop" src="https://github.com/jhjh-kim/IOT-YOLO-Project/assets/132179356/7bef0bd1-6986-4fe8-ab63-b417d07f0fe0">

To reflect various types, sizes, and angles of boats, we conducted a total of four transfer learning stages using different datasets and parameters.



### 6. Final Model Performance
![100aerial1_50boat_video_200joint_400joint](https://github.com/jhjh-kim/IOT-YOLO-Project/assets/132179356/c8120f6d-8333-4e91-85f6-c8b2db2b8b79)



### 7. Application
The drone operator sets the position to move the drone and selects the mode based on either an aerial or side view
 - video1 : aerial view + dynamic movements 
 - video2 : aerial view + multiple boats
 - video3 : oblique view 

Depending on the position and angle of the drone set, the opponent and our territorial waters are separated by red lines
- Green bbox(Normal State) :: Not over our territorial waters + Direction of progress not towards territorial waters
- Yellow bbox(Approching) :: Direction of progress towards territorial waters
- Red bbox(Crossed) :: Over our territorial waters
- Blue line :: Predicted movement direction of the boat



## USING SCIPRT FILES

### SCRIPT FILES EXPLANATION
- loss_tests :: Training results metrics for each of the four loss functions
- model_size_test :: Training results metrics for YOLOv8m and YOLOv8s on the aerial1 dataset
- ship_detector :: Model and Implementation Code for the Application
- training_results :: Training results metrics for the total of four models up to the final model



### Custom Training
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



### Ship_Detector Executation
0. Downlaod best_model.pt & ship_detector.py from 'ship_detector' Script Files

1. Download Videos from Google Drive
https://drive.google.com/drive/folders/1ETkO5a3iNDmgbmBCXtDy2OGK1IWTmSWP
   
2. Installing Libraries in the ship_detector File
===================================
pip install ultralytics
pip install opencv-python
pip install numpy
pip install torch
pip install deep_sort_realtime
===================================

3. Configuring Paths for the Model and Video

4. Executing
- video 1 ::
   - Right of the red line: Our territorial waters
   - Left of the red line: Foreign territorial waters

- video 2 ::
  - Right of the red line: Foreign territorial waters
  - Left of the red line: Our territorial waters

- video 3 :: 
   - Above of the red line: Our territorial waters
   - Below of the red line: Foreign territorial waters

## REFRENCES
1. DATASETS
- Aerial 1&2 : https://universe.roboflow.com/public-1tecx/boat-detection-bzj6m/dataset/1 
- Boat 1 : https://universe.roboflow.com/boats-hu0jt/boat-tracking-model/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true
- Boat 2 : https://universe.roboflow.com/shipdataset3-3prli/shipdata-a4ucj/dataset/9

  
2. VIDEOS
- video 1 : ==== 첨부필요 ====
- video 2 : ==== 첨부필요 ====
- video 3 : ==== 첨부필요 ====

3. CODES
- https://docs.ultralytics.com/
