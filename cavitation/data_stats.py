import datetime
import os

import matplotlib.pyplot as plt
import numpy as np

from cavitation.data.get_data import get_data
from cavitation.logger.easy_logger import get_logger


pumps = [(5, 3), (5, 18), (5, 36), (32, 3), (64, 2), (95, 2), (125, 2)]    # (flow, stages)


def main():
    logger = get_logger(__name__)

    time_tag = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = "results/data/{}".format(time_tag)

    # nothing really matters :)
    (train_m, train_x, train_y), (test_m, test_x, test_y) = get_data("fft", "regression", None, None, 0.01, flat_features=True, normalize=False, random_seed=42)

    # combine the train and test data
    data_m = np.concatenate((train_m, test_m))
    del train_m, train_x, train_y, test_m, test_x, test_y

    plt.clf()
    plt.figure(figsize=(6, 4))

    # plot a bar chart of how many records are there for each pump
    pump_counts = [0] * len(pumps)
    for record_m in data_m:
        for i, pump in enumerate(pumps):
            if pump == record_m["pump"]:
                pump_counts[i] += 1
                break

    plt.bar(range(len(pumps)), pump_counts, edgecolor="black", width=0.4, color="skyblue", zorder=3)
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7, zorder=0)
    plt.title("Number of records for each pump")
    plt.xticks(range(len(pumps)), ["{}-{}".format(flow, stages) for flow, stages in pumps], rotation=45)
    plt.xlabel("Pump (flow-stages)")
    plt.ylabel("Number of records")
    plt.tight_layout()

    logger.info("Saving the experiment figures in the directory: {}".format(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    fig_name = "pump_counts.png"
    plt.savefig(os.path.join(save_dir, fig_name), dpi=300)

    # plot the distribution of CFs for the whole dataset
    plt.clf()
    plt.figure(figsize=(6, 4))
    log_cfs = [np.log(m['CF']) for m in data_m]
    plt.hist(log_cfs, bins=20, edgecolor="black", color="skyblue", zorder=3)
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7, zorder=0)
    plt.title("Distribution of CFs")
    plt.xlabel("log(CF)")
    plt.ylabel("Number of records")
    plt.tight_layout()

    fig_name = "cf_distribution.png"
    plt.savefig(os.path.join(save_dir, fig_name), dpi=300)

    plt.close("all")    # close all figure windows (not showing anything)


if __name__ == "__main__":
    main()
