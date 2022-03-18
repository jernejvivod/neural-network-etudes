import glob
import os
import pickle as pkl
import re

import matplotlib.pyplot as plt


def plot_losses(res_file_paths, legend_values):
    """
    Plot losses parsed from results files.

    Args:
        res_file_paths (list): list of paths to results files (.res)
        legend_values (list): list of strings for the plot legend
    """

    # check if directory for plots exists. If not, create.
    plots_dir = os.path.join(os.path.dirname(__file__), 'plots/')
    os.makedirs(plots_dir, exist_ok=True)

    # set results file name
    existing_plot_files = glob.glob(plots_dir + '*.png')
    file_num_suffix = 1
    if len(existing_plot_files) > 0:
        file_num_suffix = max(map(lambda x: int(re.search(r'\d+', x).group()), existing_plot_files)) + 1
    plot_file_name = os.path.join(plots_dir, 'res' + str(file_num_suffix) + '.png')

    for path in res_file_paths:
        with open(path, 'rb') as f:
            res_nxt = pkl.load(f)
            plt.plot(range(len(res_nxt['epoch_losses'])), res_nxt['epoch_losses'])

    plt.legend(legend_values)
    plt.ylabel("Cross-entropy loss")
    plt.xlabel("Epoch")
    plt.savefig(plot_file_name)
