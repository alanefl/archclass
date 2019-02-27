#!/usr/bin/env bash

# Launch all transfer learning experiments, one after another.

python search_hyperparams.py --parent_dir experiments/t_inception_resnet_v2 --model inception_resnet_v2

python search_hyperparams.py --parent_dir experiments/t_inception_v3 --model inception_v3

python search_hyperparams.py --parent_dir experiments/t_mobilenet_v2 --model mobilenet_v2_140_224

python search_hyperparams.py --parent_dir experiments/t_nasnet_large --model nasnet_large

python search_hyperparams.py --parent_dir experiments/t_resnet_v2_152 --model resnet_v2_152
