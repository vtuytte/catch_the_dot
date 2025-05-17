from source import red_dot, index_detector
import cv2
import numpy as np

def run_singleplayer(VIDEOSIZE = (1920, 1080), WINDOWSIZE = (1920, 1080), WINDOWNAME = 'Dotgame', PADDING = 200):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, VIDEOSIZE[0])
    cap.set(4, VIDEOSIZE[1])

    # Initialize the window
    cv2.namedWindow(WINDOWNAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOWNAME, WINDOWSIZE[0], WINDOWSIZE[1])

    # Initialize the index detector
    detector = index_detector(model_path='hand_landmarker.task',
                                num_hands=1,
                                min_det_conf=0.3,
                                min_hand_conf=0.3,
                                min_track_conf=0.3)

    # Initialize the red dot
    dot = red_dot(pos=(VIDEOSIZE[0] // 2, VIDEOSIZE[1] // 2), size=50)

    # Tack number of dots popped
    num_dots = 0

    # Main program loop
    while cap.isOpened():
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Shut down if the webcam is not available
        if not ret:
            print("Failed to grab frame, shutting down...")
            break

        # Shut down if window is closed or 'q' is pressed
        if cv2.waitKey(1) == ord('q') or cv2.getWindowProperty(WINDOWNAME, cv2.WND_PROP_VISIBLE) <1:
            break
        
        # Detect hands in the frame
        detector.detect_async(frame)
        index_pos = detector.get_index_finger_pos_single()

        # Check if the index finger position is detected
        if index_pos is not None:
            # Get positions in pixel coordinates
            x, y = int(index_pos[0] * VIDEOSIZE[0]), int(index_pos[1] * VIDEOSIZE[1])
            # Draw the index finger position
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

            # Check if the red dot is close to the index finger position
            if dot.is_close((x, y)):
                # Increment the number of dots popped
                num_dots += 1
                # Move the red dot to a new random position
                new_x = np.random.randint(PADDING, VIDEOSIZE[0] - PADDING)
                new_y = np.random.randint(PADDING, VIDEOSIZE[1] - PADDING)
                dot.move((new_x, new_y))

        # Top left corner text
        cv2.putText(frame, f'Dots popped: {num_dots}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Draw finger position and red dot
        dot.draw(frame)

        # Render the frame
        cv2.imshow(WINDOWNAME, frame)

    # Close the detector, webcam, and window
    detector.close()
    cap.release()
    cv2.destroyAllWindows()

def run_multiplayer(VIDEOSIZE = (1920, 1080), WINDOWSIZE = (1920, 1080), WINDOWNAME = 'Dotgame', PADDING = 200):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, VIDEOSIZE[0])
    cap.set(4, VIDEOSIZE[1])

    # Initialize the window
    cv2.namedWindow(WINDOWNAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOWNAME, WINDOWSIZE[0], WINDOWSIZE[1])

    # Initialize the index detector
    detector = index_detector(model_path='hand_landmarker.task',
                                num_hands=2,
                                min_det_conf=0.2,
                                min_hand_conf=0.2,
                                min_track_conf=0.2)

    # Initialize the red dot
    dot = red_dot(pos=(VIDEOSIZE[0] // 2, VIDEOSIZE[1] // 2), size=50)

    # Tack number of dots popped
    num_dots_left = 0
    num_dots_right = 0
    left_color = (255, 0, 0) # Blue color
    right_color = (0, 255, 0) # Green color

    # Main program loop
    while cap.isOpened():
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Shut down if the webcam is not available
        if not ret:
            print("Failed to grab frame, shutting down...")
            break

        # Shut down if window is closed or 'q' is pressed
        if cv2.waitKey(1) == ord('q') or cv2.getWindowProperty(WINDOWNAME, cv2.WND_PROP_VISIBLE) <1:
            break
        
        # Detect hands in the frame
        detector.detect_async(frame)
        index_pos = detector.get_index_finger_pos_multi()

        # Check if the index finger position is detected
        if index_pos is not None:
            for (x, y, handedness) in index_pos:
                # Get positions in pixel coordinates
                x, y = int(x * VIDEOSIZE[0]), int(y * VIDEOSIZE[1])
                # Draw the index finger position
                if handedness == 0:
                    # Left hand
                    cv2.circle(frame, (x, y), 10, left_color, -1)
                else:
                    # Right hand
                    cv2.circle(frame, (x, y), 10, right_color, -1)

                # Check if the red dot is close to the index finger position
                if dot.is_close((x, y)):
                    # Increment the number of dots popped
                    if handedness == 0:
                        num_dots_left += 1
                    else:
                        num_dots_right += 1
                    # Move the red dot to a new random position
                    new_x = np.random.randint(PADDING, VIDEOSIZE[0] - PADDING)
                    new_y = np.random.randint(PADDING, VIDEOSIZE[1] - PADDING)
                    dot.move((new_x, new_y))

        # Top left corner text
        cv2.putText(frame, f'Dots popped by LEFT team: {num_dots_left}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f'Dots popped by RIGHT team: {num_dots_right}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Draw finger position and red dot
        dot.draw(frame)

        # Render the frame
        cv2.imshow(WINDOWNAME, frame)

    # Close the detector, webcam, and window
    detector.close()
    cap.release()
    cv2.destroyAllWindows()