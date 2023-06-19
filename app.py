import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import math

import cv2 as cv
import numpy as np
import mediapipe as mp

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    cap_device = 1
    cap_width = 960
    cap_height = 540

    use_static_image_mode = True
    min_detection_confidence = 0.7
    min_tracking_confidence = 0.5

    max_num_hands=2

    use_brect = True

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    history_length = 16
    point_history = deque(maxlen=history_length)
    rectangle_history = deque(maxlen=history_length)

    while True:
        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)

                # Hand sign classification, --> trackeia o movimento do indicador

                point_history.append(landmark_list[9])
                rectangle_history.append(brect)

                # Finger gesture classification --> Classifica o movimento do indicador
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
        else:
            point_history.append([0, 0])
            rectangle_history.append([0,0,0,0])

        debug_image = draw_point_history(debug_image, rectangle_history)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history

def draw_finger_point(image, points):
    for i in range(len(points[:-1])):
        cv.line(image, tuple(points[i]), tuple(points[i+1]),
                (255, 255, 255), 2)
    return image

def draw_palm(image, points, v1, v2):
    draw_finger_point(image, [points[v1], points[v2]])
    return image

def draw_4_fingers(image, points):
    for i in range(0, len(points), 4):
        image = draw_finger_point(image, points[i:i+4])
    return image

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        image = draw_finger_point(image, landmark_point[2:5])
        image = draw_4_fingers(image, landmark_point[5:21])
        
        image = draw_finger_point(image, landmark_point[0:3])

        image = draw_palm(image, landmark_point, 2, 5)
        image = draw_palm(image, landmark_point, 5, 9)
        image = draw_palm(image, landmark_point, 9, 13)
        image = draw_palm(image, landmark_point, 13, 17)
        image = draw_palm(image, landmark_point, 17, 0)

    # Key Points
    for landmark in landmark_point:
        cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)


    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image

def draw_point_history(image, rectangle_history):
    for index, point in enumerate(list(rectangle_history)[0:-1:4]):
        count0 = 0
        for pixel in point:
            if pixel == 0:
                count0 += 1
        if count0 == 4:
            continue
        wvalue = (point[2]-point[0])/(index+1)
        hvalue = (point[3]-point[1])/(index+1)
        width = round(wvalue)
        height = round(hvalue)
        cv.rectangle(image, (int(round(point[0]+width/1.5)), int(round(point[1]+height/1.5))), (int(round(point[2]-width/1.5)), int(round(point[3]-height/1.5))),
                     (152, 251, 152), 1)

    return image


if __name__ == '__main__':
    main()
