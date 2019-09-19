import keras
from keras.engine.input_layer import Input
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, ReLU
from keras.models import Model
import numpy as np


def thrs(x):
    return 0.0 if x < 0.5 else 1.0


def network_structure(x, y, conv_kernel_size=(3, 3),
                      print_model_structure=True):
    input_tensor = Input(x[0].shape)

    pool = input_tensor
    for maxpool in [True, False] * 3:
        conv = BatchNormalization()(Conv2D(
            32, conv_kernel_size, padding='same', activation='relu',
            kernel_initializer=keras.initializers.glorot_normal(1))(pool))
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
