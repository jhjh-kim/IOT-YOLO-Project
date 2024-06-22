import cv2
from ultralytics import YOLO
import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectTracker:
    def __init__(self, line_gps_info: tuple, side_info: int):
        """
        Initialize the object tracker.
        
        Parameters:
        line_gps_info (tuple): Tuple containing line coordinates ((x1, y1), (x2, y2))
        side_info (int): Direction information (0 = north, 1 = west, 2 = south, 3 = east)
        """
        self.line_info = line_gps_info
        self.side_info = side_info
        self.obj_status = {}
        self.track_history = {}
        self.tracker = DeepSort(max_age=30, n_init=3, max_iou_distance=0.7) 
        self.prediction_length = 10  # Number of frames to predict
        self.prev_gray = None  # To store the previous frame in grayscale

    def update(self, results, frame):
        """
        Update the tracker with the current frame and detection results.
        
        Parameters:
        results: Detection results from the YOLO model
        frame: Current video frame
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return

        # Estimate camera movement and adjust line coordinates
        self.adjust_line_coordinates(gray)

        if results and results[0]:
            bboxes = np.array(results[0].boxes.xyxy.cpu(), dtype="int")
            classes = np.array(results[0].boxes.cls.cpu(), dtype="int")
            confidences = np.array(results[0].boxes.conf.cpu(), dtype="float")
            names = results[0].names  # Get names from results

            # Prepare detections for DeepSORT
            detections = []
            for bbox, cls, conf in zip(bboxes, classes, confidences):
                if conf > 0.6:  # Filter detections with confidence > 0.6
                    detections.append((bbox, conf, cls))

            # Update tracker with new detections
            tracks = self.tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()  # left, top, right, bottom bounding box
                bbox = [int(coord) for coord in ltrb]
                (x, y, x2, y2) = bbox
                x2 -= x
                y2 -= y

                # Update track history
                center = (int((x + x2) / 2), int((y + y2) / 2))
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                self.track_history[track_id].append(center)
                history = self.track_history[track_id]
                
                if len(history) < 2:
                    continue

                pred_points = self.draw_prediction(frame, history)
                self.update_status(center, pred_points, track_id)

                # Draw bounding box, tracking ID, and status
                if self.obj_status[track_id] == 0:
                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, f"ID: {track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, f"Too Close", (x, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                elif self.obj_status[track_id] == 1:
                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, f"ID: {track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv2.putText(frame, f"Approaching", (x, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                elif self.obj_status[track_id] == 2:
                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, f"ID: {track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(frame, f"Crossed", (x, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Draw line according to line_gps_info
            self.draw_limit_line(frame)

        cv2.imshow("Img", frame)
        self.prev_gray = gray

    def adjust_line_coordinates(self, gray):
        """
        Adjust the line coordinates based on camera movement.
        
        Parameters:
        gray: Grayscale version of the current frame
        """
        # Parameters for ShiTomasi corner detection
        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        # Parameters for Lucas-Kanade optical flow
        lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Detect good features to track
        p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **feature_params)
        if p0 is None:
            return

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **lk_params)
        if p1 is None:
            return

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Compute the movement vector
        movement = np.mean(good_new - good_old, axis=0)

        # Use a threshold to determine significant camera movement
        threshold = 12.0 
        if np.linalg.norm(movement) > threshold:
            # Update line coordinates by shifting them in the opposite direction of the camera movement
            x_shift, y_shift = -movement
            x1, y1 = self.line_info[0]
            x2, y2 = self.line_info[1]
            self.line_info = ((int(x1 + x_shift), int(y1 + y_shift)), (int(x2 + x_shift), int(y2 + y_shift)))

    def draw_limit_line(self, frame):
        """
        Draw the line indicating the boundary for tracking status.
        
        Parameters:
        frame: Current video frame
        """
        (x1, y1), (x2, y2) = self.line_info
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)

    def draw_prediction(self, frame, history):
        """
        Draw the prediction path of the object.
        
        Parameters:
        frame: Current video frame
        history: List of past center positions of the object
        
        Returns:
        tuple: Start and end points of the predicted path
        """
        # Calculate the average direction of movement using the last few positions
        num_points = min(5, len(history))  # Use up to the last 5 points
        dx_sum, dy_sum = 0, 0

        for i in range(-num_points, -1):
            dx_sum += history[i+1][0] - history[i][0]
            dy_sum += history[i+1][1] - history[i][1]

        dx = dx_sum / num_points
        dy = dy_sum / num_points

        # Extrapolate the future path
        future_points = []
        for i in range(1, self.prediction_length + 1):
            start_point = (int(history[-1][0] + dx * (i - 1)), int(history[-1][1] + dy * (i - 1)))
            end_point = (int(history[-1][0] + dx * i), int(history[-1][1] + dy * i))

            cv2.line(frame, start_point, end_point, (255, 0, 0), 4, lineType=cv2.LINE_AA)
            future_points.append(end_point)
            if i == self.prediction_length:
                return (start_point, end_point)
    
    def update_status(self, center:tuple, pred_points, obj_id:int):
        """
        Update the status of the object based on its predicted path.
        
        Parameters:
        center (tuple): Current center position of the object
        pred_points (tuple): Start and end points of the predicted path
        obj_id (int): ID of the tracked object
        """
        x_diff, y_diff = ((pred_points[1][0] - pred_points[0][0]), (pred_points[1][1] - pred_points[0][1]))
        if self.side_info == 0:
            limit = (self.line_info[0][1] + self.line_info[1][1]) / 2
            if center[1] < limit:
                self.obj_status[obj_id] = 2
                return
            if y_diff < 0:
                self.obj_status[obj_id] = 1
            else:
                self.obj_status[obj_id] = 0

        elif self.side_info == 2:
            limit = (self.line_info[0][1] + self.line_info[1][1]) / 2
            if center[1] > self.line_info[0][1]:
                self.obj_status[obj_id] = 2
                return
            if y_diff > 0:
                self.obj_status[obj_id] = 1
            else:
                self.obj_status[obj_id] = 0

        elif self.side_info == 1:
            limit = (self.line_info[0][0] + self.line_info[1][0]) / 2
            if center[0] < limit:
                self.obj_status[obj_id] = 2
                return
            if x_diff < 0:
                self.obj_status[obj_id] = 1
            else:
                self.obj_status[obj_id] = 0

        elif self.side_info == 3:
            limit = (self.line_info[0][0] + self.line_info[1][0]) / 2
            if center[0] > limit:
                self.obj_status[obj_id] = 2
                return
            if x_diff > 0:
                self.obj_status[obj_id] = 1
            else:
                self.obj_status[obj_id] = 0
    
if __name__ == '__main__':
    # Load the YOLO model
    model = YOLO("/Users/jinhokim/Desktop/IOT_FINAL/400more.pt")

    #################################### video1 ####################################
    # video_path = '/Users/jinhokim/Desktop/DeepSort/final1.mp4'
    # line_gps_info1 = ((640, 0), (640, 720))
    # side_info1 = 3
    # object_tracker = ObjectTracker(line_gps_info1, side_info1)
    #################################### video1 ####################################

    #################################### video2 ####################################
    # video_path = '/Users/jinhokim/Desktop/DeepSort/final2.mp4'
    # line_gps_info2 = ((1600, 0), (1600, 2160))
    # side_info2 = 1
    # object_tracker = ObjectTracker(line_gps_info2, side_info2)
    #################################### video2 ####################################

    #################################### video3 ####################################
    video_path = '/Users/jinhokim/Desktop/DeepSort/final3.mp4'
    line_gps_info3 = ((0, 540), (1920, 540))
    side_info3 = 0
    object_tracker = ObjectTracker(line_gps_info3, side_info3)
    #################################### video3 ####################################

    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get detection results from the YOLO model
        results = model(frame)

        # For Apple Silicon
        if torch.backends.mps.is_available():
            results = model(frame, device="mps")  # Use MPS
        else:
            results = model(frame)
        
        # Update the tracker with results
        object_tracker.update(results, frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
