"""Peform hyperparameters search"""

import argparse
import os
from subprocess import check_call
import sys

from constants import MODELS
from model.utils import Params


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--job_name', required=True,
                    help="Name of this exploration")
parser.add_argument('--model', required=True,
                    choices=MODELS,
                    help="Name of the model used")
parser.add_argument('--parent_dir',
                    required=True,
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='data/prepared_arc_dataset',
                    help="Directory containing the dataset")


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

    ########
    #TODO: more user-friendly way of specifying parameter names and values
    #TODO: folder structure so we can compare specific parameter tests
    #TODO: more variety of metrics and plotting them (like confusion matrix, specific misses etc.)
    ########

    # Perform hypersearch over one parameter
    learning_rates = [1e-4, 1e-3, 1e-2]

    #random search: load from another JSON?
    #while-true loop, randomly sample hyperparam values
    #command line: name of input file

    for learning_rate in learning_rates:
        # Modify the relevant parameter in params
        params.learning_rate = learning_rate

        # Launch job (name has to be unique)
        job_name = args.job_name + "_{}".format(learning_rate)
        launch_training_job(args.parent_dir, args.data_dir, args.model, job_name, params)
