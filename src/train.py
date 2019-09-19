import numpy as np
import keras
from sklearn.metrics import accuracy_score
import cv2 as cv

from src.utils.datahelpers import split_data, load_labels, load_imgs
from src.models.net import network_structure


def main():
    root = 'data/genki4k/'
    imgs = load_imgs(root)
    labels = load_labels(root)
    X_train, X_test, y_train, y_test = split_data(imgs, labels)
    X_train, X_test, y_train, y_test = [np.array(arr, dtype=np.float32) for arr in [X_train, X_test, y_train, y_test]]
    model = network_structure(X_train, y_train)
    # model = keras.applications.MobileNetV2(classes=2, weights=None)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model.fit(X_train, y_train, epochs=3, batch_size=16)
    preds = model.predict(X_test)
    preds = [0 if score < 0.5 else 1 for score in preds]
    print(accuracy_score(preds, y_test))


if __name__ == '__main__':
    main()
