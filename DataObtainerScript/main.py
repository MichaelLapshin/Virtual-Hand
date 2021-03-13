"""
[main.py]
@description: This script is used to record and store data about hand limbs and sensor readings for the server machine learning model to use.
@author: Michael Lapshin
    - Some mediapipe code was taken from https://google.github.io/mediapipe/solutions/hands for capturing and processing the video feed.
"""

import numpy
import cv2
import mediapipe as mp


def normalized_3d_vector(v):
    return numpy.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def dot_product_3d_vector(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


# r1, r2, s are the coordinates of the 3 points.
# r2 is the point in between the reference point r1 and the point for which we are trying to interpret the angle s
# @return, angle in radians: angle is bound by [0, pi]
def coord2angle(r1, r2, s):
    reference_vector = [r2[0] - r1[0], r2[1] - r1[1], r2[2] - r1[2]]
    direction_vector = [r1[0] - s[0], r1[1] - s[1], r1[2] - s[2]]

    # Math calculations
    reference_norm = normalized_3d_vector(reference_vector)
    direction_norm = normalized_3d_vector(direction_vector)
    reference_dot_direction = dot_product_3d_vector(reference_vector, direction_vector)

    # Compute final calculations
    angle = numpy.arccos(reference_dot_direction / (reference_norm * direction_norm))
    return angle


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 10)  # Sets FPS of the video feed to 10 FPS

# Resolution set to be equal in order to keep trigonometry more consistent
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Processes and displays video feed
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.95, min_tracking_confidence=0.1,
                    static_image_mode=False) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            print(results.multi_hand_landmarks[0])

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
