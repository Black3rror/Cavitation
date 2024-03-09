import subprocess
import time


experiments = [
    # default
    {
        "name": "default",
    },

    # different input data
    {
        "name": "Experiment 1.1 - data inclusion",
        "data_include": ["raw"],
    },
    {
        "name": "Experiment 1.2 - data inclusion",
        "data_include": ["percentile"],
    },
    {
        "name": "Experiment 1.3 - data inclusion",
        "data_include": ["energy"],
    },
    {
        "name": "Experiment 1.4 - data inclusion",
        "data_include": ["std"],
    },
    {
        "name": "Experiment 1.5 - data inclusion",
        "data_include": ["percentile", "energy", "std"],
    },
    {
        "name": "Experiment 1.6 - data inclusion",
        "data_include": ["raw", "percentile", "energy", "std"],
    },

    # different window size
    {
        "name": "Experiment 2.1 - window size",
        "window_size": 256,
    },
    {
        "name": "Experiment 2.2 - window size",
        "window_size": 1024,
    },
    {
        "name": "Experiment 2.3 - window size",
        "window_size": 4096,
    },
    {
        "name": "Experiment 2.4 - window size",
        "window_size": 16384,
    },
    {
        "name": "Experiment 2.5 - window size",
        "window_size": 65536,
    },
    {
        "name": "Experiment 2.6 - window size",
        "window_size": 262144,
    },

    # different partition counts
    {
        "name": "Experiment 3.1 - data partitions",
        "n_fft_partitions": "null",
    },
    {
        "name": "Experiment 3.2 - data partitions",
        "n_fft_partitions": 5,
    },
    {
        "name": "Experiment 3.3 - data partitions",
        "n_fft_partitions": 10,
    },
    {
        "name": "Experiment 3.4 - data partitions",
        "n_fft_partitions": 25,
    },
    {
        "name": "Experiment 3.5 - data partitions",
        "n_fft_partitions": 50,
    },
    {
        "name": "Experiment 3.6 - data partitions",
        "n_fft_partitions": 100,
    },
]


def main():
    for i, exp in enumerate(experiments):
        title = "Experiment {}/{}".format(i+1, len(experiments))
        print("\n")
        print("="*80)
        print("-"*((80-len(title)-2)//2), end=" ")
        print(title, end=" ")
        print("-"*((80-len(title)-2)//2))
        print("="*80)
        if "name" in exp:
            print("Running: {}".format(exp["name"]))

        command = ["python", "-m", "cavitation.svm_model"]
        for k, v in exp.items():
            if k == "name":
                command.append("+{}={}".format(k, v))
            else:
                command.append("{}={}".format(k, v))

        tic = time.time()
        subprocess.run(command)
        toc = time.time()

        print("Experiment took {:.2f} seconds to finish".format(toc-tic))


if __name__ == "__main__":
    main()
