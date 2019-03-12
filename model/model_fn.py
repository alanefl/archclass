"""Define the model."""

import tensorflow as tf
import tf_metrics

from constants import NUM_CLASSES

from model.multinomial_logistic_regression.model import build_multinomial_logistic_regression_model
from model.basic_cnn.model import build_basic_cnn_model
"""Import your own here!"""

def model_fn(mode, inputs, params, model, reuse=False):
    """Model function defining the graph operations.

    The general model takes images and labels, and outputs a Tensor of 25 logits over all
    the possible architectural styles.

    Args:
        mode: (string) can be 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        model: (string) what model to use for this particular run.
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    labels = inputs['labels']
    labels = tf.cast(labels, tf.int64)

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):

        # Compute the output distribution of model requested and the predictions
        if model == "multinomial-logistic-regression":
            logits = build_multinomial_logistic_regression_model(
                inputs, params, reuse=reuse, is_training=is_training
            )
        elif model == "cnn-baseline":
            logits = build_basic_cnn_model(
                inputs, params, reuse=reuse, is_training=is_training
            )
            """Add your own model here!"""
        else:
            raise ValueError("Unsupported model name: %s" % model)

        predictions = tf.argmax(logits, 1)

    # Define loss and accuracy
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        if params.use_batch_norm:

            # Add a dependency to update the moving mean and variance for batch normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)

    with tf.variable_scope("metrics"):
        num_classes = NUM_CLASSES
        average = 'weighted'
        class_count = tf.bincount(tf.cast(labels, tf.int32))
        confusion = tf.confusion_matrix(labels=labels, predictions=tf.argmax(logits, 1), num_classes=num_classes)

        labels_oh = tf.one_hot(labels, num_classes, dtype=tf.int32)
        predictions_oh = tf.one_hot(predictions, num_classes, dtype=tf.int32)

        metrics = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, 1)),
            'loss': tf.metrics.mean(loss),
            'precision': tf_metrics.precision(labels, tf.argmax(logits, 1), num_classes=num_classes, average=average),
            'recall': tf_metrics.recall(labels, tf.argmax(logits, 1), num_classes=num_classes, average=average),
            'f1': tf_metrics.f1(labels, tf.argmax(logits, 1), num_classes=num_classes, average=average),
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.image('train_image', inputs['images'])

    mask = tf.not_equal(labels, predictions)

    # Add a different summary to know how they were misclassified
    for label in range(0, params.num_labels):
        mask_label = tf.logical_and(mask, tf.equal(predictions, label))
        incorrect_image_label = tf.boolean_mask(inputs['images'], mask_label)
        tf.summary.image('incorrectly_labeled_{}'.format(label), incorrect_image_label)

    bad_image = [tf.summary.image('incorrectly_labeled_{}'.format(label), tf.boolean_mask(inputs['images'], tf.logical_and(tf.not_equal(labels, predictions), tf.equal(predictions, label)))) for label in range(0, params.num_labels)]

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec['predictions'] = predictions
    model_spec['labels'] = labels
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['confusion'] = confusion
    model_spec['oh'] = tf.tuple([labels_oh, predictions_oh])
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()
    model_spec['bad_image'] = bad_image

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
