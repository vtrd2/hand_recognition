import cv2 as cv
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from collections import deque
import time
import csv
import copy
import itertools
from PIL import Image, ImageDraw
import sys, math
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

arq_name = 0

model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

def draw_point_history(image, rectangle_history):
    for index, point in enumerate(list(rectangle_history)[0:-1:3]):
        count0 = point.count(0)
        if count0 == 4:
            continue
        wvalue = (point[2] - point[0]) / (index + 1)
        hvalue = (point[3] - point[1]) / (index + 1)
        width = round(wvalue)
        height = round(hvalue)
        cv.rectangle(image, (int(round(point[0] + width / 1.5)), int(round(point[1] + height / 1.5))),
                     (int(round(point[2] - width / 1.5)), int(round(point[3] - height / 1.5))),
                     (152, 251, 152), 1)

    return image


def main():
    cap_width = 960
    cap_height = 540
    history_length = 16

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

    history = deque(maxlen=history_length)  # Initialize history with 16 points set to [0, 0, 0, 0]

    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv.flip(frame, 1)
        debug_frame = frame.copy()

        # Convert frame to RGB for Mediapipe
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Process hand detection
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                draw_landmarks(debug_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Calculate bounding box
                brect = calculate_bounding_box(hand_landmarks, frame.shape[1], frame.shape[0])

                # Draw bounding box
                cv.rectangle(debug_frame, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 2)

                # Add current palm center to history
                history.append(brect)

                landmark_list = calc_landmark_list(debug_frame, hand_landmarks)

                pre_processed_landmark_list = landmark_list

                prediction = make_prediction(pre_processed_landmark_list, brect)

                cv.putText(debug_frame, prediction, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)

        else:
            history.append([0, 0, 0, 0])

        debug_frame = draw_point_history(debug_frame, history)

        # Calcular FPS
        end_time = time.time()
        fps = 1.0 / (end_time - start_time)
        start_time = end_time

        # Exibir FPS na imagem
        cv.putText(
            debug_frame,
            f"FPS: {int(fps)}",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
            cv.LINE_AA
        )

        cv.imshow('Hand Gesture Recognition', debug_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


def calculate_bounding_box(hand_landmarks, frame_width, frame_height):
    landmark_points = [(int(landmark.x * frame_width), int(landmark.y * frame_height))
                       for landmark in hand_landmarks.landmark]

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
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

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

        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))

    return temp_point_history

def draw_lines(draw, landmark_point):
    #palma
    draw.line((landmark_point[0][0], landmark_point[0][1], landmark_point[1][0], landmark_point[1][1]), fill=(165,128,100), width=4)
    draw.line((landmark_point[1][0], landmark_point[1][1], landmark_point[2][0], landmark_point[2][1]), fill=(165,128,100), width=4)
    draw.line((landmark_point[2][0], landmark_point[2][1], landmark_point[5][0], landmark_point[5][1]), fill=(165,128,100), width=4)
    draw.line((landmark_point[5][0], landmark_point[5][1], landmark_point[9][0], landmark_point[9][1]), fill=(165,128,100), width=4)
    draw.line((landmark_point[9][0], landmark_point[9][1], landmark_point[13][0], landmark_point[13][1]), fill=(165,128,100), width=4)
    draw.line((landmark_point[13][0], landmark_point[13][1], landmark_point[17][0], landmark_point[17][1]), fill=(165,128,100), width=4)
    draw.line((landmark_point[17][0], landmark_point[17][1], landmark_point[0][0], landmark_point[0][1]), fill=(165,128,100), width=4)

    #Dedão
    draw.line((landmark_point[2][0], landmark_point[2][1], landmark_point[3][0], landmark_point[3][1]), fill=(255,28,174), width=4)
    draw.line((landmark_point[3][0], landmark_point[3][1], landmark_point[4][0], landmark_point[4][1]), fill=(255,28,174), width=4)

    # Índice do dedo
    draw.line((landmark_point[5][0], landmark_point[5][1], landmark_point[6][0], landmark_point[6][1]), fill=(100, 150, 255), width=4)
    draw.line((landmark_point[6][0], landmark_point[6][1], landmark_point[7][0], landmark_point[7][1]), fill=(100, 150, 255), width=4)
    draw.line((landmark_point[7][0], landmark_point[7][1], landmark_point[8][0], landmark_point[8][1]), fill=(100, 150, 255), width=4)

    # Dedo médio
    draw.line((landmark_point[9][0], landmark_point[9][1], landmark_point[10][0], landmark_point[10][1]), fill=(255, 255, 255), width=4)
    draw.line((landmark_point[10][0], landmark_point[10][1], landmark_point[11][0], landmark_point[11][1]), fill=(255, 255, 255), width=4)
    draw.line((landmark_point[11][0], landmark_point[11][1], landmark_point[12][0], landmark_point[12][1]), fill=(255, 255, 255), width=4)

    # Dedo anelar
    draw.line((landmark_point[13][0], landmark_point[13][1], landmark_point[14][0], landmark_point[14][1]), fill=(255, 255, 255), width=4)
    draw.line((landmark_point[14][0], landmark_point[14][1], landmark_point[15][0], landmark_point[15][1]), fill=(255, 255, 255), width=4)
    draw.line((landmark_point[15][0], landmark_point[15][1], landmark_point[16][0], landmark_point[16][1]), fill=(255, 255, 255), width=4)

    # Dedo mínimo
    draw.line((landmark_point[17][0], landmark_point[17][1], landmark_point[18][0], landmark_point[18][1]), fill=(255, 255, 0), width=4)
    draw.line((landmark_point[18][0], landmark_point[18][1], landmark_point[19][0], landmark_point[19][1]), fill=(255, 255, 0), width=4)
    draw.line((landmark_point[19][0], landmark_point[19][1], landmark_point[20][0], landmark_point[20][1]), fill=(255, 255, 0), width=4)

def adjust_landmark(landmark_list, brect):
    initx = brect[0]
    inity = brect[1]
    hand_proportion = get_hand_proportion(landmark_list)
    width = (brect[2] - brect[0])/hand_proportion
    height = (brect[3] - brect[1])/hand_proportion
    relative_width = int((width - 254) / 2)
    relative_height = int((height - 254) / 2)
    new_ll = []
    for point1, point2 in landmark_list:
        new_ll.append((((point1 - initx) / hand_proportion) - relative_width, ((point2 - inity) / hand_proportion) - relative_height))
    return new_ll

def get_hand_proportion(landmark_list):
    hand_size_base = math.sqrt((landmark_list[0][0] - landmark_list[1][0])**2 + (landmark_list[0][1] - landmark_list[1][1])**2)
    hand_proportion_base = hand_size_base / 30

    hand_size_side = math.sqrt((landmark_list[0][0] - landmark_list[17][0])**2 + (landmark_list[0][1] - landmark_list[17][1])**2)
    hand_proportion_side = hand_size_side / 64

    return hand_proportion_base if hand_proportion_base > hand_proportion_side else hand_proportion_side

def make_prediction(landmark_list, brect):
    image = Image.open("imagem.jpg")

    draw = ImageDraw.Draw(image)

    new_landmark_list = adjust_landmark(landmark_list, brect)

    draw_lines(draw, new_landmark_list)

    for point in new_landmark_list:
        draw.ellipse(((point[0] - 2), (point[1] - 2), (point[0] + 2), (point[1] + 2)), fill=(255, 0, 0))
    
    #image.save(f"mao.jpg")

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = image.convert("RGB")
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)


    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]

    # Print prediction and confidence score
    return class_name[2:-1]

if __name__ == '__main__':
    main()
