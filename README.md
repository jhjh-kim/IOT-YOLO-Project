# IOT-YOLO-Project

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
