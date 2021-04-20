# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

# General imports
import sys
import argparse

# Custom imports
sys.path.append('src')
from run_cnn import cnn_main

# ----------------------------------------------------------------------------------------------------------------------
# Program Execution
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Initializing argument parser
    parser = argparse.ArgumentParser(description="""A convolutional neural network and a Bayesian convolutional neural
                                                    network for the classification of gravitational wave signals.""")
    parser.add_argument('--model', type=str, metavar='model', help='Network model. Options are BNN and CNN.')
    parser.add_argument('--param', type=str, metavar='param', help='Path to model hyperparameters.')
    parser.add_argument('--train', type=str, metavar='train', help='Path to network training data.')
    parser.add_argument('--test', type=str, default='', metavar='test', help='Path to network testing data. Optional.')
    parser.add_argument('--res', type=str, default='results', metavar='results',
                        help='Path to where the training and validation results are saved.')
    parser.add_argument('-v', type=int, default=1, metavar='V', help='Verbosity of console output.')
    args = parser.parse_args()

    assert args.model in ('BNN', 'CNN'), 'Invalid model chosen.'

    if args.model == 'BNN':
        pass

    elif args.model == 'CNN':
        cnn_main(
            params = args.param,
            train_data = args.train,
            v = args.v,
            results_dir = args.res,
            test_data = args.test
        )

# ----------------------------------------------------------------------------------------------------------------------