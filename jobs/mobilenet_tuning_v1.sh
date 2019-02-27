#!/usr/bin/env bash

# Tuning hyperparameters on MobileNet architecture, which worked the best out of all the transfer
# learning architectures we worked on.

#
# The hyperparameters tuned are:
#    1. learning rate
#    2. batch size
#    3. number of FC layers
#    4. regularization
#    5. using batch norm or not.
#
# This differs from our previous experiment in that:
#   1) We use a different dev/train/test split
#   2) We use xavier initialization for our layers.
#
# Note that we do NOT worry about fine-tuning the pretrained architecture. We'll consider that after
# the results of this experiment. (for one, xavier initialization already seems to be giving us vastly better
# metrics).

# MobileNet trains in about 25 minutes for 100 epochs on GPU.
#
#  This means that running 100 training jobs on GPU will take about 41.6 hours.  Gonna have to use
#  random search instead of grid search for parameters.
python search_hyperparams.py --parent_dir experiments/t_mobilenet_v2 --model mobilenet_v2_140_224