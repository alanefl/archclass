"""Evaluate the model"""

import argparse
import logging
import os

import tensorflow as tf

from constants import MODEL_CHOICES
from model.input_fn import input_fn
from model.model_fn import model_fn
from model.evaluation import evaluate
from model.utils import install_tf_hub_modules
from model.utils import Params
from model.utils import set_logger
from model.utils import extract_labels


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir',
                    required=True,
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/prepared_arc_dataset',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default='best_weights',
                    help="Subdirectory of model dir or file containing the weights")
parser.add_argument('--model',
                    choices=MODEL_CHOICES,  # More models coming soon.
                    help='What model to use.',
                    required=True)
parser.add_argument('--mode',
                    choices=['confusion','bad-images','per-class'],
                    help='What metrics to return.')
parser.add_argument('--data_sub', default='test',
                    choices=['train','dev','test'],
                    help='Evaluate on train, dev, or test data.')
parser.add_argument('--parts', action='store_true', help="Pass flag if training on the 'elements' dataset.")


if __name__ == '__main__':
    # Set the random seed for the whole graph
    tf.set_random_seed(1234)

    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    data_dir = args.data_dir
    data_dir_type = args.data_sub
    test_data_dir = os.path.join(data_dir, data_dir_type)

    # Get the filenames from the test set
    test_filenames = os.listdir(test_data_dir)
    test_filenames = [os.path.join(test_data_dir, f) for f in test_filenames if f.endswith('.jpg')]

    test_labels = extract_labels(test_filenames, args.parts)

    # specify the size of the evaluation set
    params.eval_size = len(test_filenames)

    # create the iterator over the dataset
    test_inputs = input_fn(False, test_filenames, test_labels, params)

    # Install any external TF Hub module we may need
    install_tf_hub_modules(params, args.model)

    # Define the model
    logging.info("Creating the model...")
    model_spec = model_fn('eval', test_inputs, params, args.model, reuse=False)

    logging.info("Starting evaluation")
    print(args.mode)
    if args.mode == 'confusion':
        evaluate(model_spec, args.model_dir, params, args.restore_from, find_confusion=True, find_metrics=False)
    elif args.mode == 'bad-images':
        evaluate(model_spec, args.model_dir, params, args.restore_from, find_bad_images=True, find_metrics=False)
    elif args.mode == 'per-class':
        evaluate(model_spec, args.model_dir, params, args.restore_from, find_metrics=False, find_perclass_metrics=True)
    else:
        evaluate(model_spec, args.model_dir, params, args.restore_from)
