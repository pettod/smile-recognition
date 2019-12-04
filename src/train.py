"""train.py

Contains code for training and evaluating a CNN for smile detection
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, confusion_matrix

from src.utils.datahelpers import split_data, load_labels, load_imgs
from src.models.net import network_structure


def save(model_fn, weights_fn, model):
    """Save the model to a model file describing the architecture and a
    weights file containing the model weights
    """
    model_json = model.to_json()
    with open(model_fn, 'w') as f:
        f.write(model_json)
    model.save_weights(weights_fn)


def load(model_fn, weights_fn):
    """Load the model from a model file describing the architecture and a
    weights file containing the model weights
    """
    # load json and create model
    with open(model_fn, 'r') as f:
        loaded_model_json = f.read()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(weights_fn)
    return model


def main():
    """The main function: trains and evaluates the network.

    There three configuarble options in this function and they are on the first
    three lines. If a model exists in the defined path, it is loaded and
    evaluated. If it does not exist, it is first trained and then evaluated.
    """
    root = 'data/genki4k/'
    model_fn = 'data/models/net_best.json'
    weights_fn = 'data/models/weights_best.h5'

    # Load the data
    imgs = load_imgs(root)
    labels = load_labels(root)

    # Split the data
    x_train, x_test, y_train, y_test = split_data(imgs, labels)
    x_train, x_test, y_train, y_test = [
        np.array(arr, dtype=np.float32) for arr in [
            x_train, x_test, y_train, y_test]]

    # If there is a trained model at the desired path, skip training and go
    # straight to evaluation.
    if os.path.exists(model_fn) and os.path.exists(weights_fn):
        model = load(model_fn, weights_fn)
    else:
        # Load the model architecture
        model = network_structure(x_train, num_layers=4,
                                  regularization='l2')
        epochs = 50
        model.compile(
            optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

        # Train the model
        history = model.fit(
            x_train, y_train, epochs=epochs, batch_size=64,
            validation_data=(x_test, y_test))

        # Get some statistics from the training
        hist = history.history
        x_plot = list(range(1, epochs + 1))

        # Plot the statistics
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.plot(x_plot, hist['acc'], label='Train Accuracy')
        plt.plot(x_plot, hist['val_acc'], label='Validation Accuracy')
        plt.ylim(0, 1)
        plt.legend()
        plt.show()
        save(model_fn, weights_fn, model)

    # Evaluate the model
    preds = model.predict(x_test)
    preds = [0 if score < 0.5 else 1 for score in preds]
    print(confusion_matrix(y_test, preds))
    print(accuracy_score(y_test, preds))


if __name__ == '__main__':
    main()
