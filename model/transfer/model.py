"""Defines a model that uses different pretrained architectures as
as a feature extractors followed by one or two fully connected layers.

All pretrained architectures in this module come from TensorFlow hub:
    https://www.tensorflow.org/hub.

Pretrained hub modules are inserted into the params in `train.py`.

"""
import tensorflow as tf
import tensorflow_hub as hub

def build_transfer_feature_extractor_model(inputs, params, reuse=False, is_training=False):
    """Using a pretrained architecture as a feature extractor, then use one or two
    fully connected layers to generate logits.
    :return: Logits or output distribution of the model.
    """
    images = inputs['images']
    module = params.tf_hub_module['module']
    module_name = params.tf_hub_module['name']

    height, width = hub.get_expected_image_size(module)
    assert (images.get_shape()[1] == height and images.get_shape()[2] == width)
    features = module(images)

    variable_scope = "transfer_%s_feature_extractor_model" % module_name
    out = features
    with tf.variable_scope(variable_scope, reuse=reuse):
        if not params.use_single_fc:
            out = tf.layers.dense(out, params.hidden_layer_fc_sizes[0])
            out = tf.layers.dense(out, params.hidden_layer_fc_sizes[1])
        return tf.layers.dense(out, params.num_labels)
