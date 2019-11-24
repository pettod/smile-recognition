"""Static run functions"""
import numpy as np
import cv2 as cv

from src.train import load


def load_detection_model():
    model_fn = 'data/models/net_l2.json'
    weights_fn = 'data/models/weights_l2.h5'
    model = load(model_fn, weights_fn)
    return model


def main():
    filename = 'data/genki4k/files/file3110.jpg'
    model = load_detection_model()
    frame = cv.imread(filename)
    model_inpt = np.expand_dims(cv.resize(frame, (64, 64)), axis=0)
    res = model.predict(model_inpt)
    smile_status = 'Smiling' if res[0][0] > 0.3 else 'Not smiling'
    frame = cv.putText(frame, f'Smile status: {smile_status}', (0, 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv.imshow('frame', frame)
    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
