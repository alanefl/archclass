#!/usr/bin/env bash

#
# Experiments to run for reporting on final report.
#

# 1. Linear Regression (or what is this called?) w/ Xavier Initalization on its single densely connected layer.
python train.py --model_dir experiments/multinomial_logistic_regression --model multinomial-logistic-regression

# 2. Basic CNN w/ Xavier Initialization and no dropout, with image size 256.
python train.py --model_dir experiments/basic_cnn --model cnn-baseline

# 3. Transfer learning experiments, same params.json, w/ xavier initalization, a single FC layer, and 100 epochs.

python train.py --model_dir experiments/t_inception_resnet_v2 --model inception_resnet_v2
python train.py --model_dir experiments/t_inception_v3 --model inception_v3
python train.py --model_dir experiments/t_mobilenet_v2 --model mobilenet_v2_140_224
python train.py --model_dir experiments/t_nasnet_large --model nasnet_large
python train.py --model_dir experiments/t_resnet_v2_152 --model resnet_v2_152

# 4. Run the single best result from the mobilenet hyperparameter tuning.
python train.py --model_dir experiments/mobilenet_best_wk8 --model mobilenet_v2_140_224