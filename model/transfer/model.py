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
    bn_momentum = params.bn_momentum

    height, width = hub.get_expected_image_size(module)
    assert (images.get_shape()[1] == height and images.get_shape()[2] == width)
    features = module(images)

    variable_scope = "transfer_%s_feature_extractor_model" % module_name
    out = features
    xavier_initializer = tf.contrib.layers.xavier_initializer()

    weights_regularizer = tf.contrib.layers.l1_l2_regularizer(
        scale_l1=params.l1_regularization,
        scale_l2=params.l2_regularization
    )

    with tf.variable_scope(variable_scope, reuse=reuse):

        # Do a number of intermediate FC layers.
        for i in range(params.num_fc_layers - 1):
            out = tf.layers.dense(
                out,
                params.hidden_layer_fc_sizes[i],
                activation=tf.nn.relu,
                kernel_initializer=xavier_initializer,
                kernel_regularizer=weights_regularizer
            )
            if params.use_batch_norm:
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)

        # Final FC layer that outputs labels.
        return tf.layers.dense(
            out,
            params.num_labels,
            kernel_initializer=xavier_initializer,
            kernel_regularizer=weights_regularizer
        )
