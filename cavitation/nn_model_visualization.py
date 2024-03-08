import logging
import os
from io import StringIO

import hydra
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from cavitation.data.get_data import get_data
from cavitation.logger.easy_logger import get_logger


pumps = [(5, 3), (5, 18), (5, 36), (32, 3), (64, 2), (95, 2), (125, 2)]    # (flow, stages)
freq = 48000


def regression_loss(target, output):
    target = tf.math.sigmoid(4*target)
    output = tf.math.sigmoid(4*output)
    return tf.reduce_mean(tf.math.square(target - output))


def regression_accuracy(target, output):
    target_binary = tf.math.greater(target, 0)
    output_binary = tf.math.greater(output, 0)
    return tf.reduce_mean(tf.cast(tf.math.equal(target_binary, output_binary), tf.float32))


@hydra.main(config_path="configs", config_name="nn_model_visualization_config", version_base=None)
def main(cfg):
    logger = get_logger(__name__)

    (train_m, train_x, train_y), (test_m, test_x, test_y) = get_data(cfg.data_type, cfg.problem_type, cfg.window_size, cfg.test_sep_strategy, cfg.test_ratio, cfg.flat_features, normalize=True, random_seed=cfg.random_seed)
    model = tf.keras.models.load_model(cfg.model_path, custom_objects={"regression_loss": regression_loss, "regression_accuracy": regression_accuracy})

    # model summary
    logger.info("model summary")
    with StringIO() as buf:
        model.summary(print_fn=lambda x: buf.write(x + '\n'))
        summary_str = buf.getvalue()
    logger.info(summary_str)

    logger.info("evaluating the model:")
    metrics = model.evaluate(test_x, test_y, verbose=0)
    for i, metric in enumerate(model.metrics_names):
        logger.info("{}: {:.6f}".format(metric, metrics[i]))

    # show the prediction results in figures

    filenames = []
    one_sample_indeces = []
    for i in range(len(train_m)):
        if train_m[i]['record_filename'] not in filenames:
            filenames.append(train_m[i]['record_filename'])
            one_sample_indeces.append(i)
    one_sample_m = [train_m[one_sample_indeces]]
    one_sample_x = [train_x[one_sample_indeces]]
    one_sample_y = [train_y[one_sample_indeces]]
    for i in range(len(one_sample_m[0])):
        one_sample_m[0][i]['in_trainset'] = True

    one_sample_indeces = []
    for i in range(len(test_m)):
        if test_m[i]['record_filename'] not in filenames:
            filenames.append(test_m[i]['record_filename'])
            one_sample_indeces.append(i)
    if one_sample_indeces:
        one_sample_m.append(test_m[one_sample_indeces])
        one_sample_x.append(test_x[one_sample_indeces])
        one_sample_y.append(test_y[one_sample_indeces])
        for i in range(len(one_sample_m[1])):
            one_sample_m[1][i]['in_trainset'] = False

    one_sample_m = np.concatenate(one_sample_m)
    one_sample_x = np.concatenate(one_sample_x)
    one_sample_y = np.concatenate(one_sample_y)
    one_sample_predictions = model.predict(one_sample_x)
    if cfg.problem_type == 'regression':
        # we want one_sample_predictions to be probabilities, but the regression model outputs log(CF)
        # we'll use the sigmoid function to convert log(CF) to probabilities
        one_sample_predictions = 1 / (1 + np.exp(-4 * one_sample_predictions))
    one_sample_predictions = np.squeeze(one_sample_predictions)

    # get the maximum number of records for a pump (will be used to help plotting)
    n_pump_records_max = 0
    for pump in pumps:
        pump_indices = [i for i in range(len(one_sample_m)) if one_sample_m[i]['pump'] == pump]
        if len(pump_indices) > n_pump_records_max:
            n_pump_records_max = len(pump_indices)

    plt.clf()
    fig1, axes1 = plt.subplots(len(pumps), 1, figsize=(1*10, len(pumps)*6))
    fig2, axes2 = plt.subplots(len(pumps), n_pump_records_max, figsize=(n_pump_records_max*6, len(pumps)*4))

    for i, pump in enumerate(pumps):
        pump_indices = [i for i in range(len(one_sample_m)) if one_sample_m[i]['pump'] == pump]
        pump_one_sample_m = one_sample_m[pump_indices]
        pump_one_sample_x = one_sample_x[pump_indices]
        pump_one_sample_y = one_sample_y[pump_indices]
        pump_one_sample_predictions = one_sample_predictions[pump_indices]

        # sort by pump_one_sample_m['CF'], largest to smallest
        pump_CFs = [pump_one_sample_m[i]['CF'] for i in range(len(pump_one_sample_m))]
        pump_sorted_indices = np.argsort(pump_CFs)[::-1]
        pump_CFs = np.array(pump_CFs)[pump_sorted_indices]
        pump_one_sample_m = pump_one_sample_m[pump_sorted_indices]
        pump_one_sample_x = pump_one_sample_x[pump_sorted_indices]
        pump_one_sample_y = pump_one_sample_y[pump_sorted_indices]
        pump_one_sample_predictions = pump_one_sample_predictions[pump_sorted_indices]

        # plot a figure that has CF on the x-axis and prediction on the y-axis
        ax = axes1[i]
        point_colors = ['tab:red' if ((x < 1 and y > 0.5) or (x > 1 and y < 0.5)) else 'tab:blue' for x, y in zip(pump_CFs, pump_one_sample_predictions)]
        for j in range(len(pump_CFs)):
            if pump_one_sample_m[j]['in_trainset']:
                ax.scatter(np.log(pump_CFs[j]), pump_one_sample_predictions[j], color=point_colors[j], marker='o')
            else:
                ax.scatter(np.log(pump_CFs[j]), pump_one_sample_predictions[j], color=point_colors[j], marker='o', facecolors='none')
        if i == len(pumps)-1:
            ax.set_xlabel('log(CF)')
        ax.set_ylabel('Prediction (probability)')
        ax.set_title("Pump {pump_flow}-{pump_stages}".format(pump_flow=pump[0], pump_stages=pump[1]))
        ax.axvline(x=0, color='tab:gray', linestyle='--')
        ax.axhline(y=0.5, color='tab:gray', linestyle='--')
        ax.grid(True)

        # plot the records with the details on the title
        plt_val_lim = 1.05 * np.max(np.abs(one_sample_x))
        for j in range(len(pump_one_sample_m)):
            pump_record = np.squeeze(pump_one_sample_x[j])
            time = np.arange(0, len(pump_record)/freq, 1/freq)
            ax = axes2[i][j]
            ax.plot(time, pump_record)
            ax.set_ylim([-plt_val_lim, plt_val_lim])
            prediction = pump_one_sample_predictions[j]
            if cfg.problem_type == 'regression':
                # if regression, write down the CF value as prediction instead of the probability
                prediction = np.clip(pump_one_sample_predictions[j], 0.01, 0.99)
                prediction = np.exp(np.log(prediction / (1-prediction)) / 4)
            if (pump_one_sample_m[j]['CF'] >= 1 and pump_one_sample_predictions[j] >= 0.5) or (pump_one_sample_m[j]['CF'] < 1 and pump_one_sample_predictions[j] < 0.5):    # correct prediction
                ax.set_title("CF: {CF:.2f}, Prediction: {prediction:.2f}, Trained on: {trained_on}".format(CF=pump_one_sample_m[j]['CF'], prediction=prediction, trained_on="Yes" if pump_one_sample_m[j]['in_trainset'] else "No"))
            else:   # incorrect prediction
                ax.set_title("CF: {CF:.2f}, Prediction: {prediction:.2f}, Trained on: {trained_on}".format(CF=pump_one_sample_m[j]['CF'], prediction=prediction, trained_on="Yes" if pump_one_sample_m[j]['in_trainset'] else "No"), color='red')

            # setting xlabel and ylabel for each row and column
            if j == 0:
                ax.set_ylabel("Pump {pump_flow}-{pump_stages}\nAcceleration (mm/s^2)".format(pump_flow=pump[0], pump_stages=pump[1]))
            if i == len(pumps)-1:
                ax.set_xlabel("Time (s)")

    logger.info("Saving the figures in the directory: {}".format(cfg.figures_save_dir))

    os.makedirs(cfg.figures_save_dir, exist_ok=True)
    fig1.savefig(os.path.join(cfg.figures_save_dir, "predictions_summary.png"), dpi = 100)
    fig2.savefig(os.path.join(cfg.figures_save_dir, "predictions_detailed.png"), dpi = 100)

    plt.close('all')    # close all figure windows (not showing anything)


if __name__ == "__main__":
    # Info: environment variable 'TF_CPP_MIN_LOG_LEVEL' has been set to '2' in the Makefile `setup_project` target
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    os.environ["WANDB_SILENT"] = "true"

    main()
