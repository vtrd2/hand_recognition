import cv2 as cv
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import draw_landmarks

def main():
    cap_width = 960
    cap_height = 540

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

        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                draw_landmarks(
                    debug_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv.imshow('Hand Gesture Recognition', debug_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()