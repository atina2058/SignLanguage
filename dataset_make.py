import mediapipe as mp
import cv2 as cv
import copy
from components import get_xy, process_landmark, prepare_csv

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

    mode = 1
    number = -1
    while True:
        key = cv.waitKey(10)

        # exit if pressed ESC key:
        if key == 27:  # ESC
            break

        if 48 <= key <= 57:  # 0 ~ 9, also for making labels
            number = key - 48
        else:
            number = key

        print(number)
        ret, image = cap.read()

        if not ret:
            break
        image = cv.flip(image, 1)
        temp_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        # cv.imshow("results",results)
        # print(type(results))
        # print(results)

        if results.multi_hand_landmarks is not None:

            for hand_landmarks in results.multi_hand_landmarks :
                landmark_list = get_xy(temp_image, hand_landmarks)
                processed_landmark_list = process_landmark(landmark_list)
                prepare_csv(number, mode, processed_landmark_list)

                # drawing landmarks on hands:
                mp_drawing.draw_landmarks(
                    temp_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec = drawingSpec,
                    connection_drawing_spec = drawingSpec)

                info_text = " Press from 0-9 for a-j:"
                cv.putText(temp_image, info_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)
                cv.imshow('Dataset Preparation', temp_image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()


