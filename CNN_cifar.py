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


class cnn_cif:

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
        self.history
        self.model = None
        self._num_conv_layers = len(conv_filters)
        self._build()

    def summary(self):
        self.model.summary()

    def _build(self):
        self._build_cnn()

    def history(self):
        val_loss = self.model.history.history['val_loss']
        val_accuracy = self.model.history.history['val_accuracy']
        loss = self.model.history.history['loss']
        accuracy = self.model.history.history['accuracy']
        return loss, accuracy, val_loss, val_accuracy

    def compile(self, learning_rate):
        starter_learning_rate = 0.009
        end_learning_rate = 0.0009
        decay_steps = 20000
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            starter_learning_rate,
            decay_steps,
            end_learning_rate,
            power=0.2)
        optimizer = Adam(learning_rate=learning_rate_fn)
        cce = sparse_categorical_crossentropy
        self.model.compile(optimizer=optimizer, loss=cce,
                           metrics=[keras.metrics.SparseCategoricalAccuracy(), 'accuracy'])

    def train(self, x_train, y_train, batch_size, num_epochs, x_test, y_test):
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        # Shuffle and slice the dataset.
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

        # Now we get a test dataset.
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_dataset = test_dataset.batch(batch_size)

        # Since the dataset already takes care of batching,
        # we don't pass a `batch_size` argument.
        # model.fit(train_dataset, epochs=3)
        self.model.fit(train_dataset,
                       epochs=num_epochs,
                       validation_data=test_dataset,
                       # validation_split=0.2
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
        x = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"cnn_conv_layer_{layer_number}"
        )(x)
        print(self.conv_kernels[layer_index])

        # x = conv_layer(x)
        if layer_number % 2 == 0: x = SpatialDropout2D(0.3, name=f"Spatial_dropout_layer_{layer_number}")(x)
        x = MaxPooling2D(pool_size=2, strides=1)(x)
        x = gelu(x)
        x = BatchNormalization(name=f"cnn_bn_{layer_number}")(x)
        return x

    def _add_fc_layers(self, x):
        x = Flatten()(x)
        x = gelu(x)
        x = Dropout(0.3, name=f"dropout_layer_fc")(x)
        x = Dense(2 * self.output_dim, name=f"fc_layer_1")(x)
        x = BatchNormalization(name=f"cnn_fn_bn")(x)
        x = Flatten()(x)
        x = gelu(x)
        x = Dense(1.75 * self.output_dim, name=f"fc_layer_2")(x)
        x = BatchNormalization(name=f"cnn_fn_bn_1")(x)
        x = Flatten()(x)
        x = gelu(x)
        x = Dense(1.5 * self.output_dim, name=f"fc_layer_3")(x)
        x = BatchNormalization(name=f"cnn_fn_bn_2")(x)
        return x

    def _add_output_layer(self, x):
        """Flatten data and add bottleneck (Dense layer)."""
        # self._shape_before_bottleneck = K.int_shape(x)[1:]
        x = Flatten()(x)
        x = Dense(self.output_dim, name="cnn_output")(x)
        x = Softmax(name=f"Softmax_layer")(x)

        return x


if __name__ == '__main__':
    cnn = cnn_cif(
        input_shape=(28, 28, 1),
        conv_filters=(16, 16, 14),
        conv_kernels=((12, 12), (3, 3), (3, 3)),

        conv_strides=(2, 1, 1),
        output_dim=10
    )
    cnn.summary()
