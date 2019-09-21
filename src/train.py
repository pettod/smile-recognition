import os

import numpy as np
from keras.models import model_from_json
from sklearn.metrics import accuracy_score
import cv2 as cv

from src.utils.datahelpers import split_data, load_labels, load_imgs
from src.models.net import network_structure


def save(model_fn, weights_fn, model):
    model_json = model.to_json()
    with open(model_fn, 'w') as f:
        f.write(model_json)
    model.save_weights(weights_fn)


def load(model_fn, weights_fn):
    # load json and create model
    with open(model_fn, 'r') as f:
        loaded_model_json = f.read()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(weights_fn)
    return model


def main():
    root = 'data/genki4k/'
    model_fn = 'data/models/net.json'
    weights_fn = 'data/models/weights.h5'
    imgs = load_imgs(root)
    labels = load_labels(root)
    X_train, X_test, y_train, y_test = split_data(imgs, labels)
    X_train, X_test, y_train, y_test = [np.array(arr, dtype=np.float32) for arr in [X_train, X_test, y_train, y_test]]
    if os.path.exists(model_fn) and os.path.exists(weights_fn):
        model = load(model_fn, weights_fn)
    else:
        model = network_structure(X_train, y_train)
        # model = keras.applications.MobileNetV2(classes=2, weights=None)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
        save(model_fn, weights_fn, model)
    preds = model.predict(X_test)
    preds = [0 if score < 0.5 else 1 for score in preds]
    print(accuracy_score(preds, y_test))


if __name__ == '__main__':
    main()
