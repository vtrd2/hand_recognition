import cv2 as cv
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from collections import deque
import time
import copy
import itertools
from PIL import Image, ImageDraw
import math
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import threading

class Config():
    def __init__(self):
        self.cap_width = 1920
        self.cap_height = 1080
        self.history_length = 16

        self.static_image_mode=False
        self.max_num_hands=1
        self.min_detection_confidence=0.7
        self.min_tracking_confidence=0.5

        self.camera = 1


class HandsImage():
    def __init__(self, config, model, class_names):
        self.config = config

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode = self.config.static_image_mode,
            max_num_hands = self.config.max_num_hands,
            min_detection_confidence = self.config.min_detection_confidence,
            min_tracking_confidence = self.config.min_tracking_confidence
        )

        self.model = model 
        self.class_names = class_names

        self.cap = self.set_video_capture()

        self.history = deque(maxlen = self.config.history_length)

        self.prediction = None

        self.set_properties()
    
    def set_properties(self):
        cv.namedWindow("Hand Gesture Recognition", cv.WINDOW_NORMAL)
        cv.setWindowProperty("Hand Gesture Recognition", cv.WND_PROP_AUTOSIZE, cv.WND_PROP_AUTOSIZE)
    
    def set_video_capture(self):
        cap = cv.VideoCapture(self.config.camera)
        cap.set(3, self.config.cap_width)
        cap.set(4, self.config.cap_height)

        return cap

    def draw_point_history(self, image, rectangle_history):
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

    def calculate_bounding_box(self, hand_landmarks, frame_width, frame_height):
        landmark_points = [(int(landmark.x * frame_width), int(landmark.y * frame_height))
                        for landmark in hand_landmarks.landmark]

        x_min = min(landmark_points, key=lambda p: p[0])[0]
        y_min = min(landmark_points, key=lambda p: p[1])[1]
        x_max = max(landmark_points, key=lambda p: p[0])[0]
        y_max = max(landmark_points, key=lambda p: p[1])[1]

        return [x_min, y_min, x_max, y_max]

    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def draw_lines(self, draw, landmark_point):
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

    def adjust_landmark(self, landmark_list, brect):
        initx = brect[0]
        inity = brect[1]
        hand_proportion = self.get_hand_proportion(landmark_list)
        width = (brect[2] - brect[0])/hand_proportion
        height = (brect[3] - brect[1])/hand_proportion
        relative_width = int((width - 254) / 2)
        relative_height = int((height - 254) / 2)
        new_ll = []
        for point1, point2 in landmark_list:
            new_ll.append((((point1 - initx) / hand_proportion) - relative_width, ((point2 - inity) / hand_proportion) - relative_height))
        return new_ll

    def get_hand_proportion(self, landmark_list):
        hand_size_base = math.sqrt((landmark_list[0][0] - landmark_list[1][0])**2 + (landmark_list[0][1] - landmark_list[1][1])**2)
        hand_proportion_base = hand_size_base / 30

        hand_size_side = math.sqrt((landmark_list[0][0] - landmark_list[17][0])**2 + (landmark_list[0][1] - landmark_list[17][1])**2)
        hand_proportion_side = hand_size_side / 64

        return hand_proportion_base if hand_proportion_base > hand_proportion_side else hand_proportion_side

    def make_prediction(self, landmark_list, brect):
        image = Image.open("imagem.jpg")

        draw = ImageDraw.Draw(image)

        new_landmark_list = self.adjust_landmark(landmark_list, brect)

        self.draw_lines(draw, new_landmark_list)

        for point in new_landmark_list:
            draw.ellipse(((point[0] - 2), (point[1] - 2), (point[0] + 2), (point[1] + 2)), fill=(255, 0, 0))
        
        image.save(f"mao.jpg")

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
        prediction = self.model.predict(data)
        index = np.argmax(prediction)
        class_name = self.class_names[index]

        confidence_score = prediction[0][index]
        if confidence_score < 0.975:
            return None
        # Print prediction and confidence score
        self.prediction = class_name[2:-1]


class App():
    def __init__(self):
        self.config = Config()

        self.model = load_model("keras_model.h5", compile=False)
        self.class_names = open("labels.txt", "r").readlines()

        self.image = HandsImage(self.config, self.model, self.class_names)
    
    def execute(self):
        start_time = time.time()

        while True:
            ret, frame = self.image.cap.read()

            if not ret:
                break

            frame = cv.flip(frame, 1)
            debug_frame = frame.copy()

            # Convert frame to RGB for Mediapipe
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # Process hand detection
            results = self.image.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks on the frame
                    draw_landmarks(debug_frame, hand_landmarks, self.image.mp_hands.HAND_CONNECTIONS)

                    # Calculate bounding box
                    brect = self.image.calculate_bounding_box(hand_landmarks, frame.shape[1], frame.shape[0])

                    # Draw bounding box
                    cv.rectangle(debug_frame, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 2)

                    # Add current palm center to history
                    self.image.history.append(brect)

                    pre_processed_landmark_list = self.image.calc_landmark_list(debug_frame, hand_landmarks)

                    threading.Thread(target = self.image.make_prediction, args=(pre_processed_landmark_list, brect)).start()

                    cv.putText(debug_frame, self.image.prediction, (brect[0] + 5, brect[1] - 4),
                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)

            else:
                self.image.history.append([0, 0, 0, 0])

            debug_frame = self.image.draw_point_history(debug_frame, self.image.history)

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

        self.image.cap.release()
        cv.destroyAllWindows()

app = App()
app.execute()