import mediapipe as mp
import cv2 as cv
import csv
import copy
from mlModel import predictSign
from components import get_xy, process_landmark


def main():

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    drawingSpec = mp.solutions.drawing_utils.DrawingSpec(color=(199, 171, 168), thickness=2, circle_radius=2)

    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    prediction = predictSign()

    # get labels from csv for prediction:
    with open('mlModel/predict/labels.csv', encoding='utf-8-sig') as f:
        labels = csv.reader(f)
        labels = [
            row[0] for row in labels
        ]

    while True:

        # exit if pressed ESC key:
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break

        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image, 1)  # flip horizontally for correct showing in screen
        temp_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False  # for faster processing
        results = hands.process(image)
        image.flags.writeable = True

        # Detect hand and make prediction:
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmark_list = get_xy(temp_image, hand_landmarks)
                processed_landmark_list = process_landmark(landmark_list)

                sign_index = prediction(processed_landmark_list)

                # drawing landmarks on hands:
                mp_drawing.draw_landmarks(
                    temp_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec = drawingSpec,
                    connection_drawing_spec = drawingSpec)

                # print(labels[sign_index])
                # writing prediction alphabet to screen:
                info_text = "Predicted Text" + ':' + labels[sign_index]
                cv.putText(temp_image, info_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1.0, (196, 255, 255), 2, cv.LINE_AA)

        cv.imshow('Hand Gesture Recognition', temp_image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
