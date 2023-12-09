import os
import sys
import logging

from copy import deepcopy

import yaml

from train import train_from_config


def get_config_for_each_fold(config):
    for fold in range(10):
        fold = str(fold)
        fold_config = deepcopy(config)
        fold_config["data"]["path"] = fold_config["data"]["path"][:-1] + fold
        fold_config["model"]["path"] = fold_config["model"]["path"][:-1] + fold
        yield config


if __name__ == "__main__":
    # logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    fn_config = sys.argv[1]
    device = sys.argv[2]
    os.environ["CUDA_VISIBLE_DEVICES"] = device  # e.g. "0,1,2"
    with open(fn_config) as f:
        config = yaml.safe_load(f)
        fold_configs = get_config_for_each_fold(config)
        for fold_idx, fold_config in enumerate(fold_configs):
            print(fold_idx, "fold")
            if fold_idx == 0:
                continue
            train_from_config(fold_config)
