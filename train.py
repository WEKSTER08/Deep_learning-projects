from CNN_cifar import CNN
from CNN_Cifar import CNN_1
from keras.datasets import mnist, cifar10
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

LEARNING_RATE = 0.0003
BATCH_SIZE = 32
EPOCHS = 10


def load_mnist():
    (x__train, y__train), (x_test, y_test) = mnist.load_data()

    x__train = x__train.astype("float32")
    y__train = y__train.astype("float32")
    # x__train = tf.expand_dims(x__train, axis=0)
    # x__train = x__train.reshape(x__train.shape)
    x_test = x_test.astype("float32")

    # x_test = tf.expand_dims(x_test, axis=0)
    # x_test = x_test.reshape(x__train.shape)

    return x__train, y__train, x_test, y_test


def load_cifar_10():
    (x__train, y__train), (x_test, y_test) = cifar10.load_data()

    x__train = x__train.astype("float32")
    y__train = y__train.astype("float32")
    # x__train = tf.expand_dims(x__train, axis=0)
    # x__train = x__train.reshape(x__train.shape)
    x_test = x_test.astype("float32")
    # x_test = tf.expand_dims(x_test, axis=0)
    # x_test = x_test.reshape(x__train.shape)
    return x__train, y__train, x_test, y_test


def train(x__train, y__train, learning_rate, batch_size, epochs):
    custom_cnn = CNN_1(
        input_shape=(28, 28, 1),
        conv_filters=(16, 8, 8, 6),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 1, 1, 1),
        output_dim=10
    )
    custom_cnn.summary()
    custom_cnn.compile(learning_rate)
    custom_cnn.train(x__train, y__train, batch_size, epochs)
    return custom_cnn


if __name__ == "__main__":
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train, _, _ = load_cifar_10()
    cnn = train(x_train, y_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    # summarize history for accuracy
    plt.plot(cnn.history['accuracy'])
    plt.plot(cnn.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig()
    # summarize history for loss
    plt.plot(cnn.history['loss'])
    plt.plot(cnn.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig()
