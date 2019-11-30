"""Run camera loop"""
import numpy as np
import cv2 as cv
from src.run import load_detection_model


SMILE_THRS = 0.3


def main():
    """The main function: run the camera loop.
    """
    # Reserve the video camera resource
    cap = cv.VideoCapture(0)
    # Set the aspect ratio similar to the training data
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    # Load a pretrained model
    model = load_detection_model()

    # Start the camera loop
    while True:
        # Read from camera
        ret, frame = cap.read()
        if not ret:
            break

        # Feed the camera feed to the model and resize it to same as in train
        model_inpt = np.expand_dims(cv.resize(frame, (64, 64)), axis=0)
        res = model.predict(model_inpt)

        # Use a threshold to decide whether the person is smiling or not
        smile_status = 'Smiling' if res[0][0] > SMILE_THRS else 'Not smiling'
        frame = cv.putText(frame, f'Smile status: {smile_status}', (50 ,50),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

        # Show the results on screen
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
