"""Tensorflow utility functions for evaluation"""

import logging
import os

from tqdm import trange
import tensorflow as tf

from model.utils import save_dict_to_json

import numpy as np
import matplotlib.pyplot as plt

def evaluate_sess(sess, model_spec, num_steps, writer=None, params=None, find_confusion=False, find_bad_images=False, find_metrics=True, find_perclass_metrics=False):
    """Train the model on `num_steps` batches.

    Args:
        sess: (tf.Session) current session
        model_spec: (dict) contains the graph operations or nodes needed for training
        num_steps: (int) train for this number of batches
        writer: (tf.summary.FileWriter) writer for summaries. Is None if we don't log anything
        params: (Params) hyperparameters
    """
    update_metrics = model_spec['update_metrics']
    eval_metrics = model_spec['metrics']
    summary_op = model_spec['summary_op']
    loss = model_spec['loss']
    bad_image = model_spec['bad_image']

    global_step = tf.train.get_global_step()

    # Load the evaluation dataset into the pipeline and initialize the metrics init op
    sess.run(model_spec['iterator_init_op'])
    sess.run(model_spec['metrics_init_op'])

    confusion = model_spec['confusion'] #comment
    confusion_matrix = np.zeros((25,25)) #comment

    accuracy_vec = model_spec['accuracy_vec']
    precision_vec = model_spec['precision_vec']
    recall_vec = model_spec['recall_vec']
    f1_vec = model_spec['f1_vec']

    if find_bad_images:
        writer = tf.summary.FileWriter(os.path.join('experiments/basic_cnn_no_dropout', 'test_summaries'), sess.graph)

    # compute metrics over the dataset
    for _i in range(num_steps):
        if find_metrics:
            sess.run(update_metrics)
        if find_confusion:
            confusion_matrix_2 = np.add(confusion_matrix, confusion.eval(session=sess))
            confusion_matrix = confusion_matrix_2
        if find_bad_images:
            _, summ = sess.run([bad_image, summary_op])
            writer.add_summary(summ)
        if find_perclass_metrics:
            accuracy_vec = tf.convert_to_tensor(accuracy_vec).eval(session=sess)
            precision_vec = tf.convert_to_tensor(precision_vec).eval(session=sess)
            recall_vec = tf.convert_to_tensor(recall_vec).eval(session=sess)
            f1_vec = tf.convert_to_tensor(f1_vec).eval(session=sess)
    if find_confusion:
        print(confusion_matrix)
        print('Total: {}'.format(np.sum(confusion_matrix)))
        print('Correct: {}'.format(np.trace(confusion_matrix)))
        print('Accuracy: {}'.format(np.trace(confusion_matrix) / np.sum(confusion_matrix)))
        print('Actual class sizes:')
        print(np.sum(confusion_matrix, axis=1))
        print('Predicted class sizes:')
        print(np.sum(confusion_matrix, axis=0))

    # Get the values of the metrics
    if find_metrics:
        metrics_values = {k: v[0] for k, v in eval_metrics.items()}
        metrics_val = sess.run(metrics_values)
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items() if k != 'confusion')
        logging.info("- Eval metrics: " + metrics_string)
        return metrics_val

    if find_confusion:
        fig, ax = plt.subplots()
        intersection_matrix = confusion_matrix
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
        ax.matshow(intersection_matrix, cmap=plt.cm.Blues)
        plt.savefig("figure.png")

    if find_perclass_metrics:
        print("Accuracy:")
        print(accuracy_vec)
        print("Precision:")
        print(precision_vec)
        print("Recall:")
        print(recall_vec)
        print("F1:")
        print(f1_vec)


def evaluate(model_spec, model_dir, params, restore_from, find_confusion=False, find_bad_images=False, find_metrics=True, find_perclass_metrics=False):
    """Evaluate the model

    Args:
        model_spec: (dict) contains the graph operations or nodes needed for evaluation
        model_dir: (string) directory containing config, weights and log
        params: (Params) contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
        restore_from: (string) directory or file containing weights to restore the graph
    """
    # Initialize tf.Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize the lookup table
        sess.run(model_spec['variable_init_op'])

        # Reload weights from the weights subdirectory
        save_path = os.path.join(model_dir, restore_from)
        if os.path.isdir(save_path):
            save_path = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, save_path)

        # Evaluate
        num_steps = (params.eval_size + params.batch_size - 1) // params.batch_size
        if find_metrics:
            metrics = evaluate_sess(sess, model_spec, num_steps, find_confusion=find_confusion, find_bad_images=find_bad_images, find_metrics=find_metrics, find_perclass_metrics=find_perclass_metrics)
            metrics_name = '_'.join(restore_from.split('/'))
            save_path = os.path.join(model_dir, "metrics_test_{}.json".format(metrics_name))
            save_dict_to_json(metrics, save_path)
        else:
            evaluate_sess(sess, model_spec, num_steps, find_confusion=find_confusion, find_bad_images=find_bad_images, find_metrics=find_metrics, find_perclass_metrics=find_perclass_metrics)
