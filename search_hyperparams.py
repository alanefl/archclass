"""Peform hyperparameters search"""

import argparse
import os
from subprocess import check_call
import sys
import random

from model.utils import Params
from constants import MODEL_CHOICES


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir',
                    required=True,
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='data/prepared_arc_dataset',
                    help="Directory containing the dataset")
parser.add_argument('--model',
                    choices=MODEL_CHOICES,
                    help='What model to use for training.',
                    required=True)

def launch_training_job(parent_dir, data_dir, model, job_name, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name

    Args:
        parent_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} train.py --model_dir {model_dir} --data_dir {data_dir} --model {model}".format(python=PYTHON,
            model_dir=model_dir, data_dir=data_dir, model=model)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Perform hypersearch here #####################################

    # Configurations that worked well: not .01 L2 regulariation, larger  learning rates,
    #  smaller batch sizes, and L2 regularization pretty high but that does not reach 0.1. Momentum doesn't seem
    # to be affecting things too much.
    learning_rates = [5e-3, 1e-3]
    batch_size = [32, 64, 128, 256]
    number_of_fc_layers = [1, 2, 3, 4]

    fc_layers_intermediate_sizes = {
        1: [],
        2: [500],
        3: [500, 200],
        4: [1000, 250, 100]
    }

    # Note: larger L1 constants do pretty bad.
    l1_regularization = [0.0, 0.00001, 0.0001]
    l2_regularization = [0.0001, 0.001, 0.01, 0.04, 0.06, 0.08]
    dropout = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]

    batch_norm = [True, False]
    bn_momentum = [.9]

    # Tried so far (lr, batch_size, num_fc_layers, l1_reg, l2_reg, batch_norm, bn_momentum)

    # Tried on week 9 (dropout)

    tried_so_far = set()

    while True:

        # Get a random sample.
        lr = random.choice(learning_rates)
        bs = random.choice(batch_size)
        num_fc = random.choice(number_of_fc_layers)
        l1_reg = random.choice(l1_regularization)
        l2_reg = random.choice(l2_regularization)
        dropout_keep_prob = random.choice(dropout)
        use_batch_norm = random.choice(batch_norm)
        bn_mom = random.choice(bn_momentum)
        params_to_try = (lr, bs, num_fc, l1_reg, l2_reg, use_batch_norm, bn_mom)
        if params_to_try in tried_so_far:
            continue

        tried_so_far.add(params_to_try)

        # Modify the relevant parameters
        params.learning_rate = lr
        params.batch_size = bs
        params.num_fc_layers = num_fc
        params.hidden_layer_fc_sizes = fc_layers_intermediate_sizes[num_fc]
        params.l1_regularization = l1_reg
        params.l2_regularization = l2_reg
        params.dropout_keep_prob = dropout_keep_prob
        params.use_batch_norm = use_batch_norm
        params.bn_momentum = bn_mom

        # Launch job (name has to be unique)
        job_name = "hp__lr={}__bs={}__fc={}__l1={}__l2={}__dp={}__bn={}__bn_mom={}".format(
            str(lr).lower(),
            str(bs).lower(),
            str(num_fc).lower(),
            str(l1_reg).lower(),
            str(l2_reg).lower(),
            str(dropout_keep_prob).lower(),
            str(use_batch_norm).lower(),
            str(bn_mom).lower(),
        )
        print(job_name)
        launch_training_job(args.parent_dir, args.data_dir, args.model, job_name, params)

    # End hyperparameter search ####################################
