"""net.py

Defines the function that generates the network architecture
"""
import keras
from keras.engine.input_layer import Input
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization
from keras.models import Model


def network_structure(x, conv_kernel_size=(3, 3),
                      print_model_structure=True, num_layers=3,
                      batch_normalization=True, regularization=None):
    """Create the network architecture.

    Parameters
    ----------
    x: iterable
        Example of input to the network
    conv_kernel_size: tuple (or iterable) of length 2
        The size of the convolutional kernel
    print_model_structure: boolean
        Flag that controls whether the model structure is printed
    num_layers: int
        Number of layers to add to the network. A layer consists of a max
        pooling layer and a convolutional layer.
    batch_normalization: bool
        Flag for adding batch normalization
    regularization: str
        Name of the regularization method
    """
    input_tensor = Input(x[0].shape)

    # Create the network architecture. Every layer has a max pooling and a
    # convolutional layer.
    pool = input_tensor
    for _ in range(num_layers):
        conv = Conv2D(
            32, conv_kernel_size, padding='same', activation='relu',
            kernel_initializer=keras.initializers.glorot_normal(1),
            kernel_regularizer=regularization)(pool)
        if batch_normalization:
            conv = BatchNormalization()(conv)
        pool = MaxPooling2D(padding='same')(conv)

    # Dense layers
    flat_1 = Flatten()(pool)
    dense_1 = Dense(128)(flat_1)
    out = Dense(1, activation='sigmoid')(dense_1)

    # Create model
    model = Model(input=input_tensor, outputs=out)
    if print_model_structure:
        model.summary()
    return model
