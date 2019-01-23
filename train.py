"""Train the models.

Models available:
    1) Logistic Regression baseline.
    2) Neural network baseline.
    2) More coming soon.

"""

import argparse
import logging
import os

import tensorflow as tf

from constants import ARCHITECTURE_STYLES
from model.input_fn import input_fn
from model.utils import Params
from model.utils import set_logger
from model.model_fn import model_fn
from model.training import train_and_evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', required=True,
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/prepared_arc_dataset',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")
parser.add_argument('--model',
                    choices=['multinomial-logistic-regression', 'cnn-baseline'],  # More models coming soon.
                    help='What model to use.',
                    required=True)


def extract_labels(filenames):
    architecture_style_to_id = {}
    for d in ARCHITECTURE_STYLES:
        architecture_style_to_id[d['name']] = d['id']
    labels = []
    for filename in filenames:
        labels.append(
            architecture_style_to_id[
                filename.split("/")[-1].split("-")[0]
            ]
        )
    return labels


if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(1234)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwriting
    model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir, "best_weights"))
    overwriting = model_dir_has_best_weights and args.restore_from is None
    # assert not overwritting, \
    #     "Weights found in model_dir, aborting to avoid overwrite. If you don't care about \
    #         overwriting, comment me out in the source code."

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    data_dir = args.data_dir
    train_data_dir = os.path.join(data_dir, "train")
    dev_data_dir = os.path.join(data_dir, "dev")

    # Get the filenames from the train and dev sets
    train_filenames = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir)
                       if f.endswith('.jpg')]
    dev_filenames = [os.path.join(dev_data_dir, f) for f in os.listdir(dev_data_dir)
                      if f.endswith('.jpg')]

    # # Labels will be between 0 and 5 included (6 classes in total)
    train_labels, dev_labels = extract_labels(train_filenames), extract_labels(dev_filenames)

    # Specify the sizes of the dataset we train on and evaluate on
    params.train_size = len(train_filenames)
    params.eval_size = len(dev_filenames)

    # Create the two iterators over the two datasets
    train_inputs = input_fn(True, train_filenames, train_labels, params)
    eval_inputs = input_fn(False, dev_filenames, dev_labels, params)

    # Define the model
    logging.info("Creating the model...")
    train_model_spec = model_fn('train', train_inputs, params, args.model)
    eval_model_spec = model_fn('eval', eval_inputs, params, args.model, reuse=True)

    # # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir, params, args.restore_from)
