from CNN_cifar import cnn_cif
from keras.datasets import mnist, cifar10
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

LEARNING_RATE = 0.0003
BATCH_SIZE = 32
EPOCHS = 50



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


def train(x__train, y__train, learning_rate, batch_size, epochs,x_test,y_test):
    custom_cnn = cnn_cif(
        # input_shape=(28, 28, 1),
        # conv_filters=(16, 16, 10, 5),
        # conv_kernels=(12, 12, 6, 3),
        # conv_strides=(1, 1, 1, 1),
        # output_dim=10
        input_shape=(32, 32, 3),
        conv_filters=(128, 86, 32, 20),
        conv_kernels=(3, 6, 6, 10),
        conv_strides=(1,1,1,1),
        output_dim=10
    )
    custom_cnn.summary()
    custom_cnn.compile(learning_rate)
    custom_cnn.train(x__train, y__train, batch_size, epochs,x_test,y_test)
    return custom_cnn


if __name__ == "__main__":
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train,x_test,y_test = load_cifar_10()
    cnn = train(x_train, y_train, LEARNING_RATE, BATCH_SIZE, EPOCHS,x_test,y_test)
    history = cnn.history()
    # print(history[1])
    # loss - 0
    # accuracy - 1
    # val_loss - 2
    # val_accuracy - 3
    # summarize history for accuracy
    plt.plot(history[1])
    plt.plot(history[3])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # plt.savefig()
    # summarize history for loss
    plt.plot(history[0])
    plt.plot(history[2])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # plt.savefig()

        # input_shape=(28, 28, 1),
        # conv_filters=(16, 32, 5, 5),
        # conv_kernels=(10, 12, 6, 6 ),
        # conv_strides=(1, 1, 1, 1, 1, 1),
        # output_dim=10
