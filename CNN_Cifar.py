import tensorflow as tf
from keras import Model
from keras import backend as K
from keras.layers import *
from keras.losses import *
from keras.optimizers import *
from keras.activations import gelu
from keras.losses import MeanSquaredError
import numpy as np
import os
import pickle as pkl


class CNN_1:

    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 output_dim):
        self.input_shape = input_shape  # [28, 28, 1]
        self.conv_filters = conv_filters  # [2, 4, 8]
        self.conv_kernels = conv_kernels  # [3, 5, 3]
        self.conv_strides = conv_strides  # [1, 2, 2]
        self.output_dim = output_dim
        self.model = None
        self._num_conv_layers = len(conv_filters)
        self._build()

    def summary(self):
        self.model.summary()

    def _build(self):
        self._build_cnn()

    def history(self):
        pass

    def compile(self, learning_rate):
        optimizer = Adam(learning_rate=learning_rate)
        cce = sparse_categorical_crossentropy
        self.model.compile(optimizer=optimizer, loss=cce, metrics=['accuracy'])

    def train(self, x_train, y_train, batch_size, num_epochs):
        self.model.fit(x_train,
                       y_train,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=True
                       )

    def _build_cnn(self):
        cnn_input = self._add_cnn_input()
        conv_layers = self._add_conv_layers(cnn_input)
        fully_connected_layers = self._add_fc_layers(conv_layers)
        output_layer = self._add_output_layer(fully_connected_layers)
        self.model = Model(cnn_input, output_layer, name="cnn")

    def _add_cnn_input(self):
        return Input(shape=self.input_shape, name="cnn_input")

    def _add_conv_layers(self, cnn_input):
        """Create all convolutional blocks in cnn."""
        x = cnn_input
        x = ZeroPadding2D(padding=(1, 1))(x)
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        """Add a convolutional block to a graph of layers, consisting of
        conv 2d + ReLU + batch normalization.
        """
        layer_number = layer_index + 1
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"cnn_conv_layer_{layer_number}"
        )
        x = Dropout(0.5, name=f"dropout_layer_{layer_number}")(x)
        x = conv_layer(x)
        if layer_index%2 ==0 : x = MaxPooling2D(pool_size=2, strides=1)(x)
        x = gelu(x)
        x = BatchNormalization(name=f"cnn_bn_{layer_number}")(x)
        return x

    def _add_fc_layers(self, x):
        x = Flatten()(x)
        x = gelu(x)
        x = Dense(100, name=f"fc_layer_1")(x)
        x = Flatten()(x)
        x = gelu(x)
        x = Dense(50, name=f"fc_layer_2")(x)
        x = Flatten()(x)
        x = gelu(x)
        x = Dense(25, name=f"fc_layer_3")(x)
        return x

    def _add_output_layer(self, x):
        """Flatten data and add bottleneck (Dense layer)."""
        # self._shape_before_bottleneck = K.int_shape(x)[1:]
        x = Flatten()(x)
        x = Dense(self.output_dim, name="cnn_output")(x)
        x = Softmax(name=f"Softmax_layer")(x)

        return x


if __name__ == '__main__':
    cnn = CNN(
        input_shape=(32, 32, 3),
        conv_filters=(64, 32, 16, 16),
        conv_kernels=(10, 8, 8, 8),
        conv_strides=(2, 2, 1, 1),
        output_dim=10
    )
    cnn.summary()
    #     input_shape=(32, 32, 3),
    #     conv_filters=(5, 10, 10, 5),
    #     conv_kernels=(6, 10, 10, 6),
    #     conv_strides=(1, 1, 1, 1, 1, 1),
    #     output_dim=10
