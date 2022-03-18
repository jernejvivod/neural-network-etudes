import argparse
import os
import sys

from aux.data_load import load_data_cifar
from aux.res_proc import save_res
from nn.config import Config
from nn.nn import Network
from results.vis import plot_losses

if __name__ == '__main__':
    # Parse arguments.
    parser = argparse.ArgumentParser()

    parser.add_argument('--visualize-results', action='store_true', help='plot results results')
    parser.add_argument('--res-file-paths', type=str, nargs='+', help='paths to results files', required='--visualize-results' in sys.argv)
    parser.add_argument('--legend-values', type=str, nargs='+', help='plot legend values', required='--visualize-results' in sys.argv)
    parser.add_argument('--sizes', type=int, nargs='+', default=[100, 100], help='neural network hidden layer sizes')
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs to perform')
    parser.add_argument('--mini-batch-size', type=int, default=2056, help='neural network mini-batch size')
    parser.add_argument('--eta', type=float, default=0.002, help='eta parameter value')
    parser.add_argument('--dropout', action='store_true', help='use dropout or not')
    parser.add_argument('--p-dropout', type=float, default=0.4, help='probability of dropout')
    parser.add_argument('--dropout-l', type=int, nargs='+', default=[False, True, False, False], help='mask for dropout')
    parser.add_argument('--l2-reg', action='store_true', help='use l2 regularization or not')
    parser.add_argument('--lbd', type=float, default=0.01, help='lambda parameter value')
    parser.add_argument('--adam', action='store_true', help='use Adam optimizer or not')
    parser.add_argument('--beta-1', type=float, default=0.9, help='beta_1 parameter value for the Adam optimizer')
    parser.add_argument('--beta-2', type=float, default=0.999, help='beta_2 parameter value for the Adam optimizer')
    parser.add_argument('--adaptive-lr', action='store_true', help='use adaptive learning rate or not')

    # Parse training and test data path
    default_data_dir = os.path.join(os.path.dirname(__file__), 'data/')
    parser.add_argument('--train-file', type=str, default=os.path.join(default_data_dir, 'train_data.pkl'), help='path to training data file')
    parser.add_argument('--test-file', type=str, default=os.path.join(default_data_dir, 'test_data.pkl'), help='path to test data file')

    # Parse arguments
    args = parser.parse_args()

    if args.visualize_results:
        plot_losses(args.res_file_paths, args.legend_values)
    else:
        # Load and preprocess training and test data.
        # train_data, train_class, test_data, test_class = load_data(TRAIN_FILE, TEST_FILE)
        train_data, train_class, test_data, test_class = load_data_cifar(args.train_file, args.test_file)

        # Initialize neural network.
        config = Config.from_args(args)
        net = Network(config)

        # Train neural network.
        epoch_losses = net.train(train_data, train_class)

        # Evaluate neural network.
        ca, vl = net.eval_network(test_data, test_class)

        # Save results
        save_res(config, epoch_losses, ca)
