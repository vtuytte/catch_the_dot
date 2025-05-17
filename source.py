import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import time

class index_detector:
    """
    index_detector is a class for detecting hand landmarks, specifically the index finger position, using MediaPipe's HandLandmarker.
    Args:
        model_path (str): Path to the hand landmark detection model file. Defaults to 'hand_landmarker.task'.
        num_hands (int): Maximum number of hands to detect. Defaults to 1.
        min_det_conf (float): Minimum confidence for hand detection. Defaults to 0.5.
        min_hand_conf (float): Minimum confidence for hand presence. Defaults to 0.5.
        min_track_conf (float): Minimum confidence for hand tracking. Defaults to 0.5.
    Attributes:
        result: Stores the latest hand landmark detection result.
        detector: The MediaPipe HandLandmarker detector instance.
    Methods:
        detect_async(frame):
            Processes an image frame asynchronously to detect hands.
        get_index_finger_pos_single():
            Returns the (x, y) position of the index finger for a single detected hand as a numpy array.
            Returns None if no hand is detected.
        get_index_finger_pos_multi():
            Returns a numpy array of (x, y, handedness) for the index finger of each detected hand.
            Handedness is 0 for left hand, 1 for right hand.
            Returns None if no hands are detected.
        close():
            Releases resources held by the detector.
    """
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
        """
        Internal callback function to update the result of hand detection. See `detect_async` for more details.
        """
        self.result = result

    def detect_async(self, frame):
        """
        Asynchronously detects hands in the given video frame using MediaPipe.
        Args:
            frame (np.ndarray): The input image frame in RGB format as a NumPy array.
        Returns:
            None
        Note:
            The detection results are handled asynchronously via callbacks or event listeners
            registered with the detector instance.
        """
        # Convert frame to mediapipe image
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Detect hands
        self.detector.detect_async(image, timestamp_ms=int(time.time() * 1000))
    
    def get_index_finger_pos_single(self):
        """
        Returns the (x, y) position of the index finger tip from the detected hand landmarks for a single player.
        This method attempts to extract the coordinates of the index finger tip (landmark 8) from the first detected hand.
        If no hands are detected or the hand landmarks are unavailable, it returns None.
        Returns:
            np.ndarray or None: A NumPy array containing the (x, y) coordinates of the index finger tip if detected,
            otherwise None.
        """
        
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
        """
        Returns the positions of the index fingers for all detected hands in the image.
        Iterates over all detected hands and extracts the (x, y) coordinates of the index finger tip (landmark 8)
        along with the handedness (0 for left hand, 1 for right hand) for each hand.
        Returns:
            np.ndarray or None: An array of shape (num_hands, 3), where each row contains [x, y, handedness]
            for a detected hand. Returns None if no hands are detected or if an AttributeError occurs.
        """
    
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
        """
        Releases resources held by the detector.
        """
        self.detector.close()

class red_dot:
    """
    Class representing a red dot object for graphical applications.
    Attributes:
        pos (tuple): The (x, y) position of the red dot. Defaults to (0, 0).
        size (int): The radius of the red dot. Defaults to 50.
        size_sq (int): The squared size of the red dot, used for distance calculations.
        color (tuple): The color of the dot in BGR format. Defaults to (0, 0, 255) for red.
    Methods:
        __init__(pos=(0, 0), size=50):
            Initializes a red dot with the given position and size.
        draw(frame):
            Draws the red dot onto the provided frame using OpenCV.
        is_close(other_pos):
            Determines if the given position is within the dot's radius using squared distance.
        move(new_pos):
            Moves the red dot to a new position.
    """
    def __init__(self, pos = (0, 0), size = 50):
        self.pos = pos
        self.size = size
        self.size_sq = size * size
        self.color = (0, 0, 255) 

    def draw(self, frame):
        """
        Draws the red dot on the provided video frame.

        Args:
            frame (numpy.ndarray): The image or video frame on which to draw the dot.
        """
        cv2.circle(frame, self.pos, self.size, self.color, -1)
    
    def is_close(self, other_pos):
        """
        Determine whether the red dot is close to a specified position.

        Args:
            other_pos: The (x, y) coordinates to compare with the red dot's position.

        Returns:
            bool: True if the squared distance between the red dot and other_pos is less than self.size_sq, False otherwise.

        Notes:
            Uses squared distance for performance reasons (avoids computing the square root).
        """
        distance_sq = sum((np.array(self.pos) - np.array(other_pos))**2)
        return distance_sq < self.size_sq

    def move(self, new_pos):
        """
        Move the red dot to a new position.

        Args:
            new_pos: The new (x, y) position for the red dot.
        """
        self.pos = new_pos