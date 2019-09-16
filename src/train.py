import numpy as np
import keras

from src.utils.datahelpers import split_data, load_labels, load_imgs
from src.models.net import network_structure


def main():
    root = 'data/genki4k/'
    imgs = load_imgs(root)
    labels = load_labels(root)
    X_train, X_test, y_train, y_test = split_data(imgs, labels)
    X_train, X_test, y_train, y_test = [np.array(arr, dtype=np.float32) for arr in [X_train, X_test, y_train, y_test]]
    model = network_structure(X_train, y_train)
    model.compile(optimizer=keras.optimizers.SGD(lr=3e-5), loss='MSE')
    model.fit(X_train, y_train)
    print(model.predict(X_test))



if __name__ == '__main__':
    main()
