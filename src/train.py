import os

import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, confusion_matrix

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
    model_fn = 'data/models/net_l2_more_layers.json'
    weights_fn = 'data/models/weights_l2_more_layers.h5'
    imgs = load_imgs(root)
    labels = load_labels(root)
    x_train, x_test, y_train, y_test = split_data(imgs, labels)
    x_train, x_test, y_train, y_test = [
        np.array(arr, dtype=np.float32) for arr in [
            x_train, x_test, y_train, y_test]]
    if os.path.exists(model_fn) and os.path.exists(weights_fn):
        model = load(model_fn, weights_fn)
    else:
        model = network_structure(x_train, y_train, num_layers=4)
        epochs = 50
        # model = keras.applications.MobileNetV2(classes=2, weights=None)
        model.compile(
            optimizer=Adam(), loss='binary_crossentropy', metrics=['acc'])
        history = model.fit(
            x_train, y_train, epochs=epochs, batch_size=32,
            validation_data=(x_test, y_test))
        hist = history.history
        print(hist)
        x_plot = list(range(1, epochs + 1))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.plot(x_plot, hist['acc'], label='acc')
        plt.plot(x_plot, hist['val_acc'], label='val_acc')
        plt.xticks(x_plot)
        plt.ylim(0, 1)
        plt.legend()
        plt.show()
        save(model_fn, weights_fn, model)
    preds = model.predict(x_test)
    preds = [0 if score < 0.5 else 1 for score in preds]
    print(confusion_matrix(y_test, preds))
    print(accuracy_score(y_test, preds))


if __name__ == '__main__':
    main()
