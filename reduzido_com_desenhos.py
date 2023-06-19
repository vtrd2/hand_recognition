import cv2 as cv
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from collections import deque
import time
import itertools
import csv
import copy

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


def main():
    cap_width = 960
    cap_height = 540
    history_length = 32

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    cap = cv.VideoCapture(1)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    history = deque(maxlen=16)  # Initialize history with 16 points set to [0, 0, 0, 0]

    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv.flip(frame, 1)
        debug_image = frame.copy()

        # Convert frame to RGB for Mediapipe
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Process hand detection
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Draw landmarks on the frame
                draw_landmarks(
                    debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Calculate bounding box
                brect = calculate_bounding_box(hand_landmarks, frame.shape[1], frame.shape[0])

                # Draw bounding box
                cv.rectangle(debug_image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 2)

                # Add current palm center to history
                history.append(brect)

                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, history)

                logging_csv(pre_processed_landmark_list,
                pre_processed_point_history_list)

        else:
            history.append([0,0,0,0])

        debug_image = draw_point_history(debug_image, history)

        # Calcular FPS
        end_time = time.time()
        fps = 1.0 / (end_time - start_time)
        start_time = end_time

        # Exibir FPS na imagem
        cv.putText(
            debug_image,
            f"FPS: {int(fps)}",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
            cv.LINE_AA
        )
        
        cv.imshow('Hand Gesture Recognition', debug_image)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


def calculate_bounding_box(hand_landmarks, frame_width, frame_height):
    landmark_points = []
    for landmark in hand_landmarks.landmark:
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        landmark_points.append((x, y))

    x_min = min(landmark_points, key=lambda p: p[0])[0]
    y_min = min(landmark_points, key=lambda p: p[1])[1]
    x_max = max(landmark_points, key=lambda p: p[0])[0]
    y_max = max(landmark_points, key=lambda p: p[1])[1]

    return [x_min, y_min, x_max, y_max]

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

def logging_csv(landmark_list, point_history_list):
    mode = 0
    number = 0
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list, *point_history_list])


if __name__ == '__main__':
    main()
