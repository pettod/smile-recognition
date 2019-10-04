"""Run camera loop"""
import numpy as np
import cv2 as cv
from src.run import load_detection_model


def main():
    cap = cv.VideoCapture(0)
    model = load_detection_model()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        model_inpt = np.expand_dims(cv.resize(frame, (64, 64)), axis=0)
        res = model.predict(model_inpt)
        smile_status = 'Smiling' if res[0][0] > 0.3 else 'Not smiling'
        frame = cv.putText(frame, f'Smile status: {smile_status}', (50 ,50),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
