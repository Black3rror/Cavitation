import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp
import yaml

from cavitation.data.get_data import get_data
from cavitation.logger.easy_logger import get_logger


pumps = [(5, 3), (5, 18), (5, 36), (32, 3), (64, 2), (95, 2), (125, 2)]    # (flow, stages)
freq = 48000


def get_spectrogram(data, window_size=1024, overlap=512, fs=freq):
    """
    Calculate the spectrogram of the input data

    Args:
        data (np.ndarray): 1D array of the data
        window_size (int): Window size in samples
        overlap (int): Number of points to overlap between segments
        fs (int): Sampling frequency in Hz
    """
    frequencies, times, spectrogram = sp.spectrogram(data, fs=fs, window='hann', nperseg=window_size, noverlap=overlap, detrend=False, scaling='density')
    return frequencies, times, spectrogram


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg):
    logger = get_logger(__name__)

    # problem_type doesn't matter, but we fix it to "regression" for reproducibility
    # test_sep_strategy is set to None in order to have samples of any kind in the trainset
    # test_ratio is set to 0.01 in order to have a very small testset
    (train_m, train_x, train_y), (test_m, test_x, test_y) = get_data(cfg.data_type, "regression", cfg.window_size, None, 0.01, flat_features=True, normalize=False, random_seed=cfg.random_seed)

    # sort the data based on the pump and then the CF
    train_m, train_x, train_y = zip(*sorted(zip(train_m, train_x, train_y), key=lambda x: (x[0]['pump'], x[0]['CF'])))

    plt.clf()
    fig = plt.figure(figsize=(cfg.n_sample_records*6, len(pumps)*4))

    for i, pump in enumerate(pumps):
        pump_indices = [j for j in range(len(train_m)) if train_m[j]['pump'] == pump]
        pump_indices = np.rint(np.arange(0, len(pump_indices), (len(pump_indices)-1)/(cfg.n_sample_records-1)))
        pump_indices = pump_indices.astype(int)
        pump_m = [train_m[j] for j in pump_indices]
        pump_x = [train_x[j] for j in pump_indices]

        # plot the records
        if cfg.data_type == "acceleration":
            plt_val_lim = 1.05 * np.max(np.abs(np.concatenate(pump_x)))
            for j in range(len(pump_x)):
                pump_record = pump_x[j]
                pump_record_90_percentile = np.percentile(np.abs(pump_record), 90)
                pump_record_energy = np.sum(np.square(pump_record))
                pump_record_std = np.std(pump_record)
                time = np.arange(0, len(pump_record)/freq, 1/freq)
                ax = fig.add_subplot(len(pumps), cfg.n_sample_records, i*cfg.n_sample_records+j+1)
                plt.plot(time, pump_record)
                ax.set_ylim([-plt_val_lim, plt_val_lim])
                plt.title("CF: {CF:.2f}, Q: {Q:.2f}, H: {H:.2f}\n90%: {percentile_90:.2f}, Energy: {energy:.2f}, Std: {std:.2f}".format(CF=pump_m[j]['CF'], Q=pump_m[j]['Q'], H=pump_m[j]['H'], percentile_90=pump_record_90_percentile, energy=pump_record_energy, std=pump_record_std))

                # setting xlabel and ylabel for each row and column
                if j == 0:
                    plt.ylabel("Pump {pump_flow}-{pump_stages}\nAcceleration (mm/s^2)".format(pump_flow=pump[0], pump_stages=pump[1]))
                if i == len(pumps)-1:
                    plt.xlabel("Time (s)")

        elif cfg.data_type == "fft":
            pump_x = [pump_record-np.min(pump_record) for pump_record in pump_x]    # remove the negative values added by normalization
            plt_val_lim = 1.05 * np.max(np.abs(np.concatenate(pump_x)))
            for j in range(len(pump_x)):
                pump_record = pump_x[j]
                freq_line = np.arange(0, freq/2, freq/len(pump_record))
                ax = fig.add_subplot(len(pumps), cfg.n_sample_records, i*cfg.n_sample_records+j+1)
                plt.plot(freq_line, pump_record[:len(freq_line)])
                ax.set_ylim([0, plt_val_lim])
                plt.title("CF: {CF:.2f}, Q: {Q:.2f}, H: {H:.2f}".format(CF=pump_m[j]['CF'], Q=pump_m[j]['Q'], H=pump_m[j]['H']))

                # setting xlabel and ylabel for each row and column
                if j == 0:
                    plt.ylabel("Pump {pump_flow}-{pump_stages}\nAmplitude".format(pump_flow=pump[0], pump_stages=pump[1]))
                if i == len(pumps)-1:
                    plt.xlabel("Frequency (Hz)")

    logger.info("Saving the experiment info and figures in the directory: {}".format(cfg.save_dir))
    experiment_info = {"Description": ""}
    experiment_info["data_type"] = cfg.data_type
    experiment_info["window_size"] = cfg.window_size
    experiment_info["random_seed"] = cfg.random_seed
    experiment_info["train_data_1st_record"] = train_m[0]
    experiment_info["test_data_1st_record"] = test_m[0]

    yaml.Dumper.ignore_aliases = lambda *args : True
    with open(os.path.join(cfg.save_dir, "experiment_info.yaml"), 'w') as f:
        yaml.dump(experiment_info, f, indent=4, sort_keys=False)

    fig_name = "record_samples.png"
    os.makedirs(cfg.save_dir, exist_ok=True)
    plt.savefig(os.path.join(cfg.save_dir, fig_name), dpi=300)

    plt.close("all")    # close all figure windows (not showing anything)


if __name__ == "__main__":
    main()
