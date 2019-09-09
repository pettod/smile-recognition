from keras.engine.input_layer import Input
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Model
import numpy as np


def network_structure(x, y, conv_kernel_size=(3, 3),
                      print_model_structure=True):
    input_tensor = Input(x[0].shape)

    # Convolution 1
    conv_1 = Conv2D(32, conv_kernel_size, padding='same')(input_tensor)
    pool_1 = MaxPooling2D(padding='same')(conv_1)

    # Convolution 2
    conv_2 = Conv2D(32, conv_kernel_size, padding='same')(pool_1)
    pool_2 = MaxPooling2D(padding='same')(conv_2)

    # Convolution 3
    conv_3 = Conv2D(32, conv_kernel_size, padding='same')(pool_2)
    pool_3 = MaxPooling2D(padding='same')(conv_3)

    # Dense
    flat_1 = Flatten()(pool_3)
    dense_1 = Dense(128)(flat_1)
    dense_2 = Dense(1)(dense_1)

    # Create model
    model = Model(input=input_tensor, outputs=dense_2)
    if print_model_structure:
        model.summary()
    return model
