from src.utils.data_mgmt import get_data
from src.utils.common import read_config
from src.utils.model import create_model
import argparse

def training(config_path):
    config=read_config(config_path)
    # print(config)
    val_size=config['params']['validation_datasize']
    (X_train, y_train), (X_val, y_val), (X_test, y_test)=get_data(val_size)

    loss_function=config['params']['loss_function']
    optimizer=config['params']['loss_function']
    metrics=config['params']['metrics']
    model=create_model(LOSS_FUNCTION=loss_function, OPTIMIZER=optimizer, METRICS=metrics)

    EPOCHS=config['params']['epochs']
    VALIDATION=(X_val, y_val)
    history=model.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION)

if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config.yaml")
    parsed_args=args.parse_args()
    training(config_path=parsed_args.config)