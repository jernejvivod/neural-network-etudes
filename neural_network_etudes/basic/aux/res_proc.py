import glob
import os
import pickle
import re


def save_res(config, epoch_losses, ca):
    # check if results directory exists. If not, create.
    res_dir = os.path.join(os.path.dirname(__file__), '../results/res_values/')
    os.makedirs(res_dir, exist_ok=True)

    # set results file name
    existing_res_files = glob.glob(res_dir + '*.res')
    file_num_suffix = 1
    if len(existing_res_files) > 0:
        file_num_suffix = max(map(lambda x: int(re.search(r'\d+', x).group()), existing_res_files)) + 1
    res_file_name = os.path.join(res_dir, 'res' + str(file_num_suffix) + '.res')

    # save results for neural network configuration
    with open(res_file_name, 'wb') as f:
        pickle.dump({'config': config, 'epoch_losses': epoch_losses, 'ca': ca}, f)
