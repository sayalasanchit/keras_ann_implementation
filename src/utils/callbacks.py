import tensorflow as tf
import os
import numpy as np
import time

def get_unique_filename(name):
    time_asc = time.asctime().replace(" ", "_").replace(":", "_")
    unique_filename = f"{name}at{time_asc}"
    return unique_filename

def get_callbacks(config, X_train):
    logs = config['logs']
    unique_name = get_unique_filename('tb_logs')
    TENSORBOARD_ROOT_LOG_DIR = os.path.join(logs['log_dir'], logs['tensorboard_logs'], unique_name)

    os.makedirs(TENSORBOARD_ROOT_LOG_DIR, exist_ok=True)
    
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_ROOT_LOG_DIR)

    file_writer=tf.summary.create_file_writer(logdir=TENSORBOARD_ROOT_LOG_DIR)
    with file_writer.as_default():
        images = np.reshape(X_train[10:30], newshape=(-1, 28, 28, 1)) # Taking to images as sample and reshaping each to 28X28X1
        tf.summary.image("20 handwritten samples", images, max_outputs=25, step=0)

    params=config['params']
    earlystopping_cb = tf.keras.callbacks.EarlyStopping(patience=params['patience'], restore_best_weights=params['restore_best_weights'])

    artifacts=config['artifacts']
    checkpoint_dir=os.path.join(artifacts['artifact_dir'], artifacts['checkpoint_dir'])
    os.makedirs(checkpoint_dir, exist_ok=True)

    CKPT_path = os.path.join(checkpoint_dir, "model_ckpt.h5")
    modelcheckpointing_cb = tf.keras.callbacks.ModelCheckpoint(CKPT_path, save_best_only=params['save_best_only'])

    return [tensorboard_cb, earlystopping_cb, modelcheckpointing_cb]
