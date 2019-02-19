#!/usr/bin/env bash

# We have to do this manually, since my computer won't validate the SSL
# certifiates of tfhub's .dev TLD.

mkdir -p hub_modules

# mobilenet_v2_140_224 Feature Vectors.
mkdir -p hub_modules/mobilenet_v2_140_224
curl -L "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/2?tf-hub-format=compressed" | tar -zxvC hub_modules/mobilenet_v2_140_224
