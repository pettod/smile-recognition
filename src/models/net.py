"""net.py

Defines the function that generates the network architecture
"""
import keras
from keras.engine.input_layer import Input
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization
from keras.models import Model


def thrs(x):
    return 0.0 if x < 0.5 else 1.0


def network_structure(x, y, conv_kernel_size=(3, 3),
                      print_model_structure=True, num_layers=3,
                      batch_normalization=True, regularization=None):
    input_tensor = Input(x[0].shape)

    # Create the network architecture. Every other layer has a max pooling
    # layer.
    pool = input_tensor
    for maxpool in [True, False] * num_layers:
        conv = Conv2D(
            32, conv_kernel_size, padding='same', activation='relu',
            kernel_initializer=keras.initializers.glorot_normal(1),
            kernel_regularizer=regularization)(pool)
        if batch_normalization:
            conv = BatchNormalization()(conv)
        if maxpool:
            pool = MaxPooling2D(padding='same')(conv)
        else:
            pool = conv

    # Dense
    flat_1 = Flatten()(pool)
    dense_1 = Dense(128)(flat_1)
    out = Dense(1, activation='sigmoid')(dense_1)

    # Create model
    model = Model(input=input_tensor, outputs=out)
    if print_model_structure:
        model.summary()
    return model
