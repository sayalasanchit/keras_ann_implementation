import tensorflow as tf


def get_data(val_size):
    mnist = tf.keras.datasets.mnist
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

    # Train validation split
    X_val, X_train = X_train_full[:val_size] / 255., X_train_full[val_size:] / 255.
    y_val, y_train = y_train_full[:val_size], y_train_full[val_size:]
    X_test = X_test / 255.

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
