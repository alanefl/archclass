#!/usr/bin/env bash

# Tuning hyperparameters on MobileNet architecture, which worked the best out of all the transfer
# learning architectures we worked on.

#
# The hyperparameters tuned are:
#    1. learning rate
#    2. batch size
#    3. number of FC layers
#    4. for each FC layer, what size to use.
#    5. regularization (TBD)
#    6. Whether to fine-tune the architecture or not.
#


# Mobilenet trains in about 25 minutes for 100 epochs on GPU.
python search_hyperparams.py --parent_dir experiments/t_mobilenet_v2 --model mobilenet_v2_140_224