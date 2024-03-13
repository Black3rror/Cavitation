import datetime
import logging
import os

import hydra
import numpy as np
import yaml
from omegaconf import OmegaConf
from sklearn import svm

from cavitation.data.get_data import get_data
from cavitation.logger.easy_logger import get_logger


pumps = [(5, 3), (5, 18), (5, 36), (32, 3), (64, 2), (95, 2), (125, 2)]    # (flow, stages)
freq = 48000


def _get_stats(data_x_main, percentile=True, energy=True, std=True, n_partitions=None):
    # data_x_main has shape (n_samples, n_features)
    data_x_extra = []   # it will take the shape (n_extra_features)(n_samples)
    if percentile:
        data_x_extra.append(np.percentile(np.abs(data_x_main), 90, axis=1))
    if energy:
        data_x_extra.append(np.sum(np.square(data_x_main), axis=1))
    if std:
        data_x_extra.append(np.std(data_x_main, axis=1))

    if n_partitions is not None:
        data_parts = np.array_split(data_x_main, n_partitions, axis=1)  # shape: (n_partitions)(n_samples, n_features/n_partitions)
        if percentile:
            data_x_extra += [np.percentile(t, 90, axis=1) for t in data_parts]
        if energy:
            data_x_extra += [np.sum(np.square(t), axis=1) for t in data_parts]
        if std:
            data_x_extra += [np.std(t, axis=1) for t in data_parts]

    if data_x_extra == []:
        data_x_extra = None
    else:
        data_x_extra = np.array(data_x_extra).T     # shape: (n_samples, n_extra_features)
    return data_x_extra


@hydra.main(config_path="configs", config_name="svm_model_config", version_base=None)
def main(cfg):
    logger = get_logger(__name__)

    cfg.time_tag = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    logger.info("Loading data ...")
    (train_m, train_x_main, train_y), (test_m, test_x_main, test_y) = get_data(cfg.data_type, "classification", cfg.window_size, cfg.test_sep_strategy, cfg.test_ratio, flat_features=True, normalize=False, random_seed=cfg.random_seed)
    logger.info("Data has been loaded")

    logger.info("Calculating pump stats ...")
    n_partitions = cfg.n_fft_partitions if cfg.data_type == "fft" else None
    include_percentile = True if "percentile" in cfg.data_include else False
    include_energy = True if "energy" in cfg.data_include else False
    include_std = True if "std" in cfg.data_include else False

    train_x_extra = _get_stats(train_x_main, include_percentile, include_energy, include_std, n_partitions)
    test_x_extra = _get_stats(test_x_main, include_percentile, include_energy, include_std, n_partitions)

    if "raw" in cfg.data_include:
        if train_x_extra is not None:
            train_x = np.concatenate((train_x_main, train_x_extra), axis=1)
            test_x = np.concatenate((test_x_main, test_x_extra), axis=1)
        else:
            train_x = train_x_main
            test_x = test_x_main
    else:
        train_x = train_x_extra
        test_x = test_x_extra
    logger.info("Pump stats have been calculated")

    logger.info("")
    pumps_accuracy = {}
    for pump in pumps:
        pump_train_indices = [i for i in range(len(train_m)) if train_m[i]['pump'] == pump]
        pump_test_indices = [i for i in range(len(test_m)) if test_m[i]['pump'] == pump]
        pump_train_x = train_x[pump_train_indices]
        pump_train_y = train_y[pump_train_indices]
        pump_test_x = test_x[pump_test_indices]
        pump_test_y = test_y[pump_test_indices]

        # normalize train_x and test_x
        pump_train_x_mean = np.mean(pump_train_x)
        pump_train_x_std = np.std(pump_train_x)
        pump_train_x = (pump_train_x - pump_train_x_mean) / pump_train_x_std
        pump_test_x = (pump_test_x - pump_train_x_mean) / pump_train_x_std

        # train the SVM model
        model = svm.LinearSVC(dual=False)
        model.fit(pump_train_x, pump_train_y)

        # evaluate the model
        pump_train_accuracy = model.score(pump_train_x, pump_train_y)
        pump_test_accuracy = model.score(pump_test_x, pump_test_y)
        logger.info("Pump {}-{}: Train accuracy: {:.6f}, Test accuracy: {:.6f}".format(pump[0], pump[1], pump_train_accuracy, pump_test_accuracy))
        pumps_accuracy["{}-{}".format(pump[0], pump[1])] = {"train_accuracy": pump_train_accuracy, "test_accuracy": pump_test_accuracy}

    average_train_accuracy = np.mean([pumps_accuracy["{}-{}".format(pump[0], pump[1])]["train_accuracy"] for pump in pumps])
    average_test_accuracy = np.mean([pumps_accuracy["{}-{}".format(pump[0], pump[1])]["test_accuracy"] for pump in pumps])
    logger.info("-" * 66)
    logger.info("Average train accuracy: {:.6f}, Average test accuracy: {:.6f}".format(average_train_accuracy, average_test_accuracy))
    logger.info("")

    data_include = OmegaConf.to_container(cfg.data_include, resolve=True)   # otherwise, saving it in the yaml file will raise an error/bug (infinite recursion)
    experiment_info = {"Description": ""}
    if "name" in cfg:
        experiment_info["Description"] = cfg.name
    experiment_info["data_type"] = cfg.data_type
    experiment_info["window_size"] = cfg.window_size
    experiment_info["test_sep_strategy"] = cfg.test_sep_strategy
    experiment_info["test_ratio"] = cfg.test_ratio
    experiment_info["random_seed"] = cfg.random_seed
    experiment_info["data_include"] = data_include
    experiment_info["n_fft_partitions"] = cfg.n_fft_partitions
    experiment_info["pumps_accuracy"] = pumps_accuracy
    experiment_info["average_train_accuracy"] = float(average_train_accuracy)
    experiment_info["average_test_accuracy"] = float(average_test_accuracy)
    experiment_info["train_data_1st_record"] = train_m[0]
    experiment_info["test_data_1st_record"] = test_m[0]

    logger.info("Saving the experiment info in the directory: {}".format(cfg.save_dir))
    os.makedirs(cfg.save_dir, exist_ok=True)
    yaml.Dumper.ignore_aliases = lambda *args : True
    with open(os.path.join(cfg.save_dir, "experiment_info.yaml"), 'w') as f:
        yaml.dump(experiment_info, f, indent=4, sort_keys=False)


if __name__ == "__main__":
    # Info: environment variable 'TF_CPP_MIN_LOG_LEVEL' has been set to '2' in the Makefile `setup_project` target
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    os.environ["WANDB_SILENT"] = "true"

    main()
