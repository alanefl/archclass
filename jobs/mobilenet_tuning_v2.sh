#!/usr/bin/env bash

# Tuning hyperparameters on MobileNet architecture, which worked the best out of all the transfer
# learning architectures we worked on.

#
# The hyperparameters tuned are:
#    1. dropout.
#
# This differs from our previous experiment (week 9) in that:
#   1) We introduce dropout
#   2) We narrow in on the parameters that worked best in the week 9 experiments.
#
# Note that we do NOT worry about fine-tuning the pretrained architecture. We can overfit the training set
# from the feature vector outputs of the pretrained architecture.
python search_hyperparams.py --parent_dir experiments/t_mobilenet_v2 --model mobilenet_v2_140_224