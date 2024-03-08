import datetime
import logging
import os

import hydra
import numpy as np
import yaml
from sklearn import svm

from cavitation.data.get_data import get_data
from cavitation.logger.easy_logger import get_logger


pumps = [(5, 3), (5, 18), (5, 36), (32, 3), (64, 2), (95, 2), (125, 2)]    # (flow, stages)
freq = 48000


def _get_stats(record, n_partitions=None):
    record_90_percentile = np.percentile(np.abs(record), 90)
    record_energy = np.sum(np.square(record))
    record_std = np.std(record)
    record_extra = [record_90_percentile, record_energy, record_std]

    if n_partitions is not None:
        record_parts = np.array_split(record, n_partitions)
        record_parts_90_percentile = [np.percentile(t, 90) for t in record_parts]
        record_parts_energy = [np.sum(np.square(t)) for t in record_parts]
        record_parts_std = [np.std(t) for t in record_parts]
        record_extra += record_parts_90_percentile + record_parts_energy + record_parts_std

    record_extra = np.array(record_extra)
    return record_extra


@hydra.main(config_path="configs", config_name="svm_model_config", version_base=None)
def main(cfg):
    logger = get_logger(__name__)

    cfg.time_tag = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    (train_m, train_x_main, train_y), (test_m, test_x_main, test_y) = get_data(cfg.data_type, "classification", cfg.window_size, cfg.test_sep_strategy, cfg.test_ratio, flat_features=True, normalize=False, random_seed=cfg.random_seed)
    train_x, test_x = None, None

    # add pump stats to train_x and test_x
    for pump in pumps:
        pump_train_indices = [i for i in range(len(train_m)) if train_m[i]['pump'] == pump]
        pump_test_indices = [i for i in range(len(test_m)) if test_m[i]['pump'] == pump]
        pump_train_x = train_x_main[pump_train_indices]
        pump_test_x = test_x_main[pump_test_indices]

        # add stats to train_x and test_x
        for i in range(len(pump_train_x)):
            pump_record = pump_train_x[i]
            n_partitions = cfg.n_fft_partitions if cfg.data_type == "fft" else None
            pump_x_extra = _get_stats(pump_record, n_partitions)
            if train_x is None:     # create for the first time
                train_x = np.zeros((train_x_main.shape[0], train_x_main.shape[1] + len(pump_x_extra)))
            train_x[pump_train_indices[i]] = np.concatenate((pump_train_x[i], pump_x_extra))

        for i in range(len(pump_test_x)):
            pump_record = pump_test_x[i]
            n_partitions = cfg.n_fft_partitions if cfg.data_type == "fft" else None
            pump_x_extra = _get_stats(pump_record, n_partitions)
            if test_x is None:      # create for the first time
                test_x = np.zeros((test_x_main.shape[0], test_x_main.shape[1] + len(pump_x_extra)))
            test_x[pump_test_indices[i]] = np.concatenate((pump_test_x[i], pump_x_extra))

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
        model = svm.SVC(kernel="linear")
        model.fit(pump_train_x, pump_train_y)

        # evaluate the model
        pump_train_accuracy = model.score(pump_train_x, pump_train_y)
        pump_test_accuracy = model.score(pump_test_x, pump_test_y)
        logger.info("Pump {}-{}: Train accuracy: {:.6f}, Test accuracy: {:.6f}".format(pump[0], pump[1], pump_train_accuracy, pump_test_accuracy))
        pumps_accuracy[pump] = {"train_accuracy": pump_train_accuracy, "test_accuracy": pump_test_accuracy}

    average_train_accuracy = np.mean([pumps_accuracy[pump]["train_accuracy"] for pump in pumps])
    average_test_accuracy = np.mean([pumps_accuracy[pump]["test_accuracy"] for pump in pumps])
    logger.info("")
    logger.info("Average train accuracy: {:.6f}, Average test accuracy: {:.6f}".format(average_train_accuracy, average_test_accuracy))

    experiment_info = {"Description": ""}
    experiment_info["data_type"] = cfg.data_type
    experiment_info["window_size"] = cfg.window_size
    experiment_info["test_sep_strategy"] = cfg.test_sep_strategy
    experiment_info["test_ratio"] = cfg.test_ratio
    experiment_info["random_seed"] = cfg.random_seed
    experiment_info["n_fft_partitions"] = cfg.n_fft_partitions
    experiment_info["pumps_accuracy"] = pumps_accuracy
    experiment_info["average_train_accuracy"] = average_train_accuracy
    experiment_info["average_test_accuracy"] = average_test_accuracy
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
