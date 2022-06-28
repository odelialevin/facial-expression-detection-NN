import numpy as np
import cv2
import os

RECT_COLOR = (255,0,0)
TEXT_COLOR = (0,0,255)
EXPRESSIONS = ["Angry", "Disgust",
                    "Fear", "Happy",
                    "Neutral", "Sad",
                    "Surprise"]
class CameraCapture(object):
    def __init__(self, model, face_detector):
        self.capture = cv2.VideoCapture(0)
        self.model = model
        self.face_detector = face_detector
    def __del__(self):
        self.capture.release()

    # returns camera frames along with bounding boxes and predictions
    def get_current_frame(self):

        a, frame = self.capture.read()

        gray_scale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(gray_scale_frame, 1.3, 5)

        for (x, y, width, height) in faces:

            face_frame = gray_scale_frame[y:y+height, x:x+width]
            face_frame = cv2.resize(face_frame, (48, 48))
            face_frame_pred_ready = face_frame[np.newaxis, :, :, np.newaxis]

            all_predictions = self.model.predict(face_frame_pred_ready, verbose=0)

            pred = EXPRESSIONS[np.argmax(all_predictions)]

            cv2.putText(frame, pred, (x, y+20), cv2.FONT_HERSHEY_COMPLEX, 1, TEXT_COLOR, 2)
            cv2.rectangle(frame,(x,y),(x+width ,y+height) ,RECT_COLOR,1)
        return frame

def main_loop(capture):
    while True:
        frame = capture.get_current_frame()
        cv2.imshow('Facial Expression Recognization',frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

def start_app(model):
    cv2_path = os.path.dirname(cv2.__file__)
    HAARCASCADE_PATH = cv2_path + '/data/haarcascade_frontalface_default.xml'
    face_detector = cv2.CascadeClassifier(HAARCASCADE_PATH)

    main_loop(CameraCapture(model, face_detector))