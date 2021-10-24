from src.utils.data_mgmt import get_data
from src.utils.common import read_config
from src.utils.model import create_model, save_model, save_plot
import os
import argparse


def training(config_path):
    config = read_config(config_path)
    # print(config)
    val_size = config['params']['validation_datasize']
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data(val_size)

    loss_function = config['params']['loss_function']
    optimizer = config['params']['optimizer']
    metrics = config['params']['metrics']
    model = create_model(LOSS_FUNCTION=loss_function, OPTIMIZER=optimizer, METRICS=metrics)

    EPOCHS = config['params']['epochs']
    VALIDATION = (X_val, y_val)
    history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION)

    artifacts_dir = config['artifacts']['artifact_dir']

    model_name = config['artifacts']['model_name']
    model_dir = config['artifacts']['model_dir']

    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)

    save_model(model, model_name, model_dir_path)

    plot_dir = config['artifacts']['plot_dir']
    plot_dir_path = os.path.join(artifacts_dir, plot_dir)
    os.makedirs(plot_dir_path, exist_ok=True)
    
    plot_name=model_name.split('.')[0]
    save_plot(history, plot_name, plot_dir_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config.yaml")
    parsed_args = args.parse_args()
    print(parsed_args.config)
    training(config_path=parsed_args.config)
