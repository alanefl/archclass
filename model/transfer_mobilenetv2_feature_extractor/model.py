"""Defines a model that uses the last layer of MobileNet V2
as a feature extractor into a linear softmax classifier.

See https://tfhub.dev/google/imagenet/mobilenet_v2_035_128/feature_vector/2.

"""
import tensorflow as tf
import tensorflow_hub as hub


def build_transfer_mobilenetv2_feature_extractor_model(inputs, params, reuse=False, is_training=False):
    """Using pretrained MobileNet V2 as a feature extractor before a softmax classifier we train.
    :return: Logits or output distribution of the model.
    """
    images = inputs['images']
    module = params.tf_hub_module
    height, width = hub.get_expected_image_size(module)
    assert (images.get_shape()[1] == height and images.get_shape()[2] == width)
    features = module(images)
    with tf.variable_scope('transfer_mobilenetv2_feature_extractor_model', reuse=reuse):
        return tf.layers.dense(features, params.num_labels)
