from src.utils.data_mgmt import get_data
from src.utils.common import read_config
import argparse

def training(config_path):
    config=read_config(config_path)
    # print(config)
    val_size=config['params']['validation_datasize']
    (X_train, y_train), (X_val, y_val), (X_test, y_test)=get_data(val_size)

if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config.yaml")
    parsed_args=args.parse_args()
    training(config_path=parsed_args.config)