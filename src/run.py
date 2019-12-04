"""run.py

Static run functions
"""
import numpy as np
import cv2 as cv

from src.train import load


SMILE_THRS = 0.3


def load_detection_model():
    """Load a detection model"""
    model_fn = 'data/models/net_best.json' #'data/models/net_l2.json'
    weights_fn = 'data/models/weights_best.h5'
    model = load(model_fn, weights_fn)
    return model


def main():
    """The main function: run single image through the model"""
    # Example images:
    # 'data/genki4k/files/file3899.jpg': not smiling
    # 'data/genki4k/files/file0621.jpg': smiling
    filename = 'data/genki4k/files/file0621.jpg'

    # Load the model
    model = load_detection_model()

    # Load the image
    frame = cv.imread(filename)

    # Feed the image though the model
    model_inpt = np.expand_dims(cv.resize(frame, (64, 64)), axis=0)
    res = model.predict(model_inpt)

    # Decide whether the person is smiling
    smile_status = 'Smiling' if res[0][0] > SMILE_THRS else 'Not smiling'
    frame = cv.putText(frame, f'Result: {smile_status}', (0, 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    # Show the results on screen
    cv.imshow('frame', frame)
    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
