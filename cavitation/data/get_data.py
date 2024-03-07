import os

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm


pumps = [(5, 3), (5, 18), (5, 36), (32, 3), (64, 2), (95, 2), (125, 2)]    # (flow, stages)
dataset_root_dir = "data"
pumps_metadata_file_path = os.path.join(dataset_root_dir, "raw/CR {pump_flow}-{pump_stages} calculations.xlsx")
pumps_metadata_sheet_name = "NPSH points"
pumps_record_file_path = os.path.join(dataset_root_dir, "raw/CR{pump_flow}-{pump_stages}/{record_file_name}")


def get_data(data_type, problem_type, window_size, test_sep_strategy, test_ratio, flat_features=True, random_seed=None):
    """
    Loads the dataset.

    Args:
        data_type (str): The data type. One of ["acceleration", "fft"].
        problem_type (str): The problem type. One of ["regression", "classification"].
        window_size (int): The window size. If None, the whole record is used as a single sample.
        test_sep_strategy (str): The test set separation strategy. One of [None, "data", "record", "pump"].
        test_ratio (float): The ratio of the test set.
        flat_features (bool): If True, the features are flattened.
        random_seed (int): The random seed. If None, the random seed is not set.

    Returns:
        tuple: ((train_m, train_x, train_y), (test_m, test_x, test_y))
    """

    cached = False
    if os.path.exists(os.path.join(dataset_root_dir, "processed/cache_info.yaml")):
        cache_info = OmegaConf.load(os.path.join(dataset_root_dir, "processed/cache_info.yaml"))
        if cache_info.data_type == data_type and cache_info.problem_type == problem_type and cache_info.window_size == window_size and cache_info.test_sep_strategy == test_sep_strategy and cache_info.test_ratio == test_ratio and cache_info.flat_features == flat_features and cache_info.random_seed == random_seed and random_seed is not None:
            cached = True

    if not cached:
        rng = np.random.RandomState(random_seed)
        window_size_original = window_size

        # collect the data
        dataset_m = []       # metadata
        dataset_x = []
        dataset_y = []
        for i in tqdm(range(len(pumps)), desc="Loading data", colour='green', leave=False):
            pump = pumps[i]

            pump_metadata = _read_pump_metadata(pump[0], pump[1], exclude_partial_data=True)
            pump_records_filenames = pump_metadata['Filename'].to_numpy()
            pump_records_Q = pump_metadata['Q'].to_numpy()
            pump_records_H = pump_metadata['H'].to_numpy()
            pump_records_CF = pump_metadata['CF'].to_numpy()

            pump_dataset_m = []
            pump_dataset_x = []
            pump_dataset_y = []
            for j in range(len(pump_records_filenames)):
                pump_record = _read_pump_record(pump[0], pump[1], pump_records_filenames[j])

                if window_size_original is None:
                    window_size = len(pump_record)

                for k in range(0, len(pump_record)-window_size+1, int(window_size/2)):
                    pump_dataset_m.append({"pump": pump,
                                        "record_filename": pump_records_filenames[j],
                                        "window_start": k,
                                        "window_end": k+window_size,
                                        "Q": pump_records_Q[j],
                                        "H": pump_records_H[j],
                                        "CF": pump_records_CF[j]})
                    pump_dataset_x.append(np.array(pump_record[k:k+window_size]))
                    pump_dataset_y.append(pump_records_CF[j])

            dataset_m.append(np.array(pump_dataset_m))
            dataset_x.append(np.array(pump_dataset_x))
            dataset_y.append(np.array(pump_dataset_y))

        dataset_m = np.concatenate(dataset_m)
        dataset_x = np.concatenate(dataset_x)
        dataset_y = np.concatenate(dataset_y)

        # process the data
        dataset_x = _convert_from_measurement_to_acceleration(dataset_x)
        if data_type == "acceleration":
            pass
        elif data_type == "fft":
            # If we don't partition the dataset, we may crash the kernel
            dataset_x = np.array_split(dataset_x, 20)
            for i in tqdm(range(len(dataset_x)), desc="Calculating FFT of data", colour='green', leave=False):
                dataset_x[i] = np.fft.fft(dataset_x[i], axis=1)
                dataset_x[i] = np.abs(dataset_x[i])
                dataset_x[i] = dataset_x[i][:, :int(window_size/2)].astype('float32')  # convert to float32 to save memory, otherwise we may crash the kernel
            dataset_x = np.concatenate(dataset_x)
        else:
            raise ValueError(f"Invalid data_type: {data_type}")

        dataset_y = np.log(dataset_y)

        if problem_type == "regression":
            pass
        elif problem_type == "classification":
            dataset_y = (dataset_y >= 0).astype(int)
        else:
            raise ValueError(f"Invalid problem_type: {problem_type}")

        dataset_x = (dataset_x - np.mean(dataset_x)) / np.std(dataset_x)

        if flat_features:
            pass
        else:
            dataset_x = np.expand_dims(dataset_x, axis=-1)    # having width of 1
            dataset_x = np.expand_dims(dataset_x, axis=-1)    # having 1 channel

        dataset_x = dataset_x.astype('float32')
        dataset_y = dataset_y.astype('float32')

        # shuffle the data
        shuffled_indices = rng.permutation(len(dataset_x))
        dataset_m = dataset_m[shuffled_indices]
        dataset_x = dataset_x[shuffled_indices]
        dataset_y = dataset_y[shuffled_indices]

        # separate the data
        if test_sep_strategy is None:
            # randomly split the dataset without respecting anything
            test_size = int(len(dataset_x)*test_ratio)
            train_m = dataset_m[test_size:]
            train_x = dataset_x[test_size:]
            train_y = dataset_y[test_size:]
            test_m = dataset_m[:test_size]
            test_x = dataset_x[:test_size]
            test_y = dataset_y[:test_size]

        elif test_sep_strategy == "data":
            # for each pump, select `test_ratio` of data points for the test set
            train_m, train_x, train_y = [], [], []
            test_m, test_x, test_y = [], [], []
            for pump in pumps:
                pump_indices = [i for i in range(len(dataset_m)) if dataset_m[i]['pump'] == pump]
                test_size = int(len(pump_indices)*test_ratio)
                train_indices = pump_indices[test_size:]
                test_indices = pump_indices[:test_size]
                train_m.append(dataset_m[train_indices])
                train_x.append(dataset_x[train_indices])
                train_y.append(dataset_y[train_indices])
                test_m.append(dataset_m[test_indices])
                test_x.append(dataset_x[test_indices])
                test_y.append(dataset_y[test_indices])
            train_m = np.concatenate(train_m)
            train_x = np.concatenate(train_x)
            train_y = np.concatenate(train_y)
            test_m = np.concatenate(test_m)
            test_x = np.concatenate(test_x)
            test_y = np.concatenate(test_y)

        elif test_sep_strategy == "record":
            # for each pump, randomly select `test_ratio` of records for the test set
            train_m, train_x, train_y = [], [], []
            test_m, test_x, test_y = [], [], []
            for pump in pumps:
                pump_indices = [i for i in range(len(dataset_m)) if dataset_m[i]['pump'] == pump]
                pump_filenames = np.unique([dataset_m[i]['record_filename'] for i in pump_indices])
                # select `test_ratio` of `pump_records_filenames` for the test set
                pump_filenames_test = rng.choice(pump_filenames, size=round(len(pump_filenames)*test_ratio), replace=False)
                pump_filenames_train = [filename for filename in pump_filenames if filename not in pump_filenames_test]
                pump_train_indices = [i for i in pump_indices if dataset_m[i]['record_filename'] in pump_filenames_train]
                pump_test_indices = [i for i in pump_indices if dataset_m[i]['record_filename'] in pump_filenames_test]
                train_m.append(dataset_m[pump_train_indices])
                train_x.append(dataset_x[pump_train_indices])
                train_y.append(dataset_y[pump_train_indices])
                test_m.append(dataset_m[pump_test_indices])
                test_x.append(dataset_x[pump_test_indices])
                test_y.append(dataset_y[pump_test_indices])
            train_m = np.concatenate(train_m)
            train_x = np.concatenate(train_x)
            train_y = np.concatenate(train_y)
            test_m = np.concatenate(test_m)
            test_x = np.concatenate(test_x)
            test_y = np.concatenate(test_y)

        elif test_sep_strategy == "pump":
            # randomly select `test_ratio` of pumps for the test set
            pumps_test_chosen = np.random.choice(len(pumps), size=round(len(pumps)*test_ratio), replace=False)
            pumps_test = [pumps[i] for i in pumps_test_chosen]
            pumps_train = [i for i in pumps if i not in pumps_test]
            pump_train_indices = [i for i in range(len(dataset_m)) if dataset_m[i]['pump'] in pumps_train]
            pump_test_indices = [i for i in range(len(dataset_m)) if dataset_m[i]['pump'] in pumps_test]
            train_m = dataset_m[pump_train_indices]
            train_x = dataset_x[pump_train_indices]
            train_y = dataset_y[pump_train_indices]
            test_m = dataset_m[pump_test_indices]
            test_x = dataset_x[pump_test_indices]
            test_y = dataset_y[pump_test_indices]

        else:
            raise ValueError(f"Invalid test_sep_strategy: {test_sep_strategy}")

        os.makedirs(os.path.join(dataset_root_dir, "processed"), exist_ok=True)
        OmegaConf.save(OmegaConf.create({"data_type": data_type, "problem_type": problem_type, "window_size": window_size_original, "test_sep_strategy": test_sep_strategy, "test_ratio": test_ratio, "flat_features": flat_features, "random_seed": random_seed}), os.path.join(dataset_root_dir, "processed/cache_info.yaml"))
        np.savez_compressed(os.path.join(dataset_root_dir, "processed/dataset.npz"), train_m=train_m, train_x=train_x, train_y=train_y, test_m=test_m, test_x=test_x, test_y=test_y)

    else:
        dataset = np.load(os.path.join(dataset_root_dir, "processed/dataset.npz"), allow_pickle=True)
        train_m = dataset['train_m']
        train_x = dataset['train_x']
        train_y = dataset['train_y']
        test_m = dataset['test_m']
        test_x = dataset['test_x']
        test_y = dataset['test_y']

    return (train_m, train_x, train_y), (test_m, test_x, test_y)


def _read_pump_metadata(pump_flow, pump_stages, exclude_partial_data=True):
    """
    Reads the pump metadata from the excel file.

    Args:
        pump_flow (int): The pump flow.
        pump_stages (int): The pump stages.
        exclude_partial_data (bool): If True, excludes the data that doesn't have both the Filename and CF values.
    """

    pump_metadata_file_path = pumps_metadata_file_path.format(pump_flow=pump_flow, pump_stages=pump_stages)
    pump_metadata = pd.read_excel(pump_metadata_file_path, sheet_name=pumps_metadata_sheet_name, header=None)

    # excel file has many irrelevant rows, so we need to strip it and only keep the useful table

    # Find the row that contains the column labels. This is the first row that contains the string 'TimeStamp'
    header_row = pump_metadata[pump_metadata.eq('TimeStamp').any(axis=1)].index[0]

    # Set the found row as column labels and exclude it from the data
    pump_metadata.columns = pump_metadata.iloc[header_row]
    pump_metadata = pump_metadata.iloc[header_row+1:]

    # Reset the row index to start from 0
    pump_metadata = pump_metadata.reset_index(drop=True)

    if exclude_partial_data:
        pump_metadata = pump_metadata.dropna(subset=['Filename', 'CF'])

    # Sort rows based on their CF value
    pump_metadata = pump_metadata.sort_values(by=['CF'], ascending=False)

    return pump_metadata


def _read_pump_record(pump_flow, pump_stages, record_file_name):
    pump_record_file_path = pumps_record_file_path.format(pump_flow=pump_flow, pump_stages=pump_stages, record_file_name=record_file_name)
    pump_record = pd.read_csv(pump_record_file_path, header=None, dtype='uint16').to_numpy().flatten()

    return pump_record


def _convert_from_measurement_to_acceleration(digi_val):
    # 12 bitnumber from digital value in GiM converted to voltage then acceleration (mm/s^2)
    # U = 3.3*digi_val/4095
    # Acc = (U-3.3/2)*1/(11.5*0.04*3.3/5)
    ACC = ((digi_val/4095)-0.5)*10.87
    return(ACC)

if __name__ == '__main__':
    pass
