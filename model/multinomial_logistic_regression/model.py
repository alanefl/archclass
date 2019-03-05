"""Defines a multinomial logistic regression baseline.
"""
import tensorflow as tf


def build_multinomial_logistic_regression_model(inputs, params, reuse=False, is_training=False):
    """Simple softmax logistic regression model.
    :return: Logits or output distribution of the model.
    """
    with tf.variable_scope('multinomial_logistic_regression', reuse=reuse):
        img_size = params.image_size
        images = inputs['images']
        num_labels = params.num_labels
        xavier_initializer = tf.contrib.layers.xavier_initializer()
        assert images.get_shape().as_list() == [None, img_size, img_size, 3]
        return tf.layers.dense(tf.layers.flatten(images), num_labels, kernel_initializer=xavier_initializer)
