import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import time

class index_detector:
    def __init__(self, model_path = 'hand_landmarker.task', num_hands = 1, min_det_conf = 0.5, min_hand_conf = 0.5, min_track_conf = 0.5):
        self.result = vision.HandLandmarkerResult

        # Initialize hand tracking model
        base_options = python.BaseOptions(model_asset_path=model_path)
        VisionRunningMode = vision.RunningMode
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                            num_hands=num_hands, 
                                            running_mode=VisionRunningMode.LIVE_STREAM,
                                            min_hand_detection_confidence=min_det_conf,
                                            min_hand_presence_confidence=min_hand_conf,
                                            min_tracking_confidence=min_track_conf,
                                            result_callback=self._internal_update)
        
        # Create hand detector
        self.detector = vision.HandLandmarker.create_from_options(options)
        
    def _internal_update(self, result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.result = result

    def detect_async(self, frame):
        # Convert frame to mediapipe image
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Detect hands
        self.detector.detect_async(image, timestamp_ms=int(time.time() * 1000))
    
    def get_index_finger_pos_single(self):
        # Return the position of the index finger in the image for single player

        # Sometimes the model has never seen a hand in that case calling results.hand_world_landmarks
        # results in an attribute error so we catch it and return None
        try:
            # Check if any hands were detected
            if not self.result.hand_world_landmarks:
                return None
        
            hand_landmarks = self.result.hand_landmarks[0] # We only care about the first hand
            index_finger = hand_landmarks[8] # We want the 8th position: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#models

            return np.array([index_finger.x, index_finger.y])
        
        except AttributeError as e:
            return None

    def get_index_finger_pos_multi(self):
        # Return the position of the index finger in the image for multiple players

        # Sometimes the model has never seen a hand in that case calling results.hand_world_landmarks
        # results in an attribute error so we catch it and return None
        try:
            # Check if any hands were detected
            if not self.result.hand_world_landmarks:
                return None

            # Loop over all detected hands and save position and handedness
            output = list()
            hand_landmarks = self.result.hand_landmarks
            for i in range(len(hand_landmarks)):
                index_finger = hand_landmarks[i][8] # We want the 8th position: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#models
                left_hand = self.result.handedness[i][0].index # 0 for left hand, 1 for right hand
                output.append([index_finger.x, index_finger.y, left_hand])
            return np.array(output)

        except AttributeError as e:
            return None

    def close(self):
        self.detector.close()

class red_dot:
    def __init__(self, pos = (0, 0), size = 50):
        self.pos = pos
        self.size = size
        self.size_sq = size * size
        self.color = (0, 0, 255) # Red color in BGR format

    def draw(self, frame):
        # Draw the red dot on the frame
        cv2.circle(frame, self.pos, self.size, self.color, -1) # -1 fills the circle
    
    def is_close(self, other_pos):
        # Check if the red dot is close to the other position
        # Using squared distance to avoid sqrt for performance
        distance_sq = sum((np.array(self.pos) - np.array(other_pos))**2)
        return distance_sq < self.size_sq

    def move(self, new_pos):
        # Move the red dot to a new position
        self.pos = new_pos