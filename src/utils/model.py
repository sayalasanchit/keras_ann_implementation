import tensorflow as tf
import time
import os


def create_model(LOSS_FUNCTION, OPTIMIZER, METRICS):
    LAYERS = [
        tf.keras.layers.Flatten(input_shape=(28, 28), name="input_layer"),
        tf.keras.layers.Dense(units=300, activation="relu", name="hidden_layer_1"),
        tf.keras.layers.Dense(units=100, activation="relu", name="hidden_layer_2"),
        tf.keras.layers.Dense(units=10, activation="softmax", name="output")
    ]
    model_clf = tf.keras.Sequential(LAYERS)

    model_clf.summary()

    model_clf.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=METRICS)
    return model_clf  # Untrained model


def get_unique_filename(filename):
    unique_filename = time.strftime(f"%Y%m%d_%H%M%S_{filename}")
    return unique_filename


def save_model(model, model_name, model_dir):
    unique_filename = get_unique_filename(model_name)
    path_to_model = os.path.join(model_dir, unique_filename)
    model.save(path_to_model)