#!/usr/bin/env bash

# We have to do this manually, since my computer won't validate the SSL
# certifiates of tfhub's .dev TLD.

mkdir -p hub_modules

# mobilenet_v2_140_224 Feature Vectors.
mkdir -p hub_modules/mobilenet_v2_140_224
curl -L "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/2?tf-hub-format=compressed" | tar -zxvC hub_modules/mobilenet_v2_140_224

# inception_v3 Feature Vectors.
mkdir -p hub_modules/inception_v3
curl -L "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1?tf-hub-format=compressed" | tar -zxvC hub_modules/inception_v3

# inception_resnet_v2 Feature Vectors.
mkdir -p hub_modules/inception_resnet_v2
curl -L "https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/1?tf-hub-format=compressed" | tar -zxvC hub_modules/inception_resnet_v2

# NASNet-A Large Feature Vectors.
mkdir -p hub_modules/nasnet_large
curl -L "https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/1?tf-hub-format=compressed" | tar -zxvC hub_modules/nasnet_large

# ResNet V2 152
mkdir -p hub_modules/resnet_v2_152
curl -L "https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/1?tf-hub-format=compressed" | tar -zxvC hub_modules/resnet_v2_152

