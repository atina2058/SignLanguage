
from flask import Flask, render_template, Response
import cv2 as cv
import mediapipe as mp
import csv
import copy
import threading
from mlModel import predictSign
from components import get_xy, process_landmark

app = Flask(__name__)

# Global variable to keep the video capture object and Mediapipe setup preloaded
camera = None
mp_hands = None
mp_drawing = None
drawingSpec = None
prediction = None
labels = None

def initialize_camera_and_mediapipe():
    global camera, mp_hands, mp_drawing, drawingSpec, prediction, labels
    if camera is None:
        camera = cv.VideoCapture(0)
        camera.set(cv.CAP_PROP_FRAME_WIDTH, 960)
        camera.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

    if mp_hands is None:
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        drawingSpec = mp.solutions.drawing_utils.DrawingSpec(color=(199, 171, 168), thickness=2, circle_radius=2)
        prediction = predictSign()

    if labels is None:
        with open('mlModel/predict/labels.csv', encoding='utf-8-sig') as f:
            labels = csv.reader(f)
            labels = [row[0] for row in labels]

def generate_frames():
    global camera, mp_hands, mp_drawing, drawingSpec, prediction, labels
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        temp_image = copy.deepcopy(frame)
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        frame_rgb.flags.writeable = False
        results = hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmark_list = get_xy(temp_image, hand_landmarks)
                processed_landmark_list = process_landmark(landmark_list)

                sign_index = prediction(processed_landmark_list)

                mp_drawing.draw_landmarks(
                    temp_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=drawingSpec,
                    connection_drawing_spec=drawingSpec
                )

                info_text = f"Predicted Text: {labels[sign_index]}"
                cv.putText(temp_image, info_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1.0, (196, 255, 255), 2, cv.LINE_AA)

        ret, buffer = cv.imencode('.jpg', temp_image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Preload camera and mediapipe on a separate thread to reduce startup time
    preload_thread = threading.Thread(target=initialize_camera_and_mediapipe)
    preload_thread.start()
    preload_thread.join()  # Ensure preload is complete before running the app

    app.run(debug=True)
