"""Tensorflow utility functions for evaluation"""

import logging
import os

from tqdm import trange
import tensorflow as tf

from model.utils import save_dict_to_json
from constants import NUM_CLASSES
from constants import ARCHITECTURE_STYLES

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def get_human_readable_label(label_oh):
    idx = 0
    for idx_, label in enumerate(label_oh):
        if label == 1:
            idx = idx_
            break

    for style in ARCHITECTURE_STYLES:
        if int(style["id"]) == idx:
            return style["name"]
    return None


def evaluate_sess(sess,
                  model_spec,
                  num_steps,
                  writer=None,
                  params=None,
                  print_all=False,
                  find_confusion=False,
                  find_bad_images=False,
                  find_metrics=True,
                  find_perclass_metrics=False):
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

    confusion = model_spec['confusion']
    confusion_matrix = np.zeros((NUM_CLASSES,NUM_CLASSES))

    oh_tensor = model_spec['oh']

    labels_oh = np.zeros((NUM_CLASSES, 0))
    predictions_oh = np.zeros((NUM_CLASSES, 0))

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
            labels_returned, predictions_returned = tf.convert_to_tensor(oh_tensor).eval(session=sess)

            # Let's stack one-hot vectors for labels and predictions horizontally over the entire dataset.
            labels_oh = np.concatenate((labels_oh, labels_returned.T), axis=1)
            predictions_oh = np.concatenate((predictions_oh, predictions_returned.T), axis=1)

    assert(predictions_oh.shape == labels_oh.shape)

    if print_all:
        _, examples = predictions_oh.shape
        for i in range(examples):
            prediction = get_human_readable_label(predictions_oh[:,i])
            label = get_human_readable_label(labels_oh[:, i])
            print(label, prediction)

    exit(0)

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
        labels_two = np.multiply(2, labels_oh)
        evaluation = np.subtract(labels_two, predictions_oh)
        #evaluation values:
        #tp = 2 - 1 = 1
        #tn = 0 - 0 = 0
        #fp = 0 - 1 = -1
        #fn = 2 - 0 = 2
        tp = np.equal(evaluation, 1)
        tn = np.equal(evaluation, 0)
        fp = np.equal(evaluation, -1)
        fn = np.equal(evaluation, 2)
        tp_int = tp.astype(float)
        tn_int = tn.astype(float)
        fp_int = fp.astype(float)
        fn_int = fn.astype(float)
        tp_count = np.sum(tp_int, axis=1)
        tn_count = np.sum(tn_int, axis=1)
        fp_count = np.sum(fp_int, axis=1)
        fn_count = np.sum(fn_int, axis=1)

        predicted_pos = np.add(tp_count, fp_count)
        true_pos = np.add(tp_count, fn_count)

        accuracy_vec = np.divide(np.add(tp_count, tn_count), np.add(true_pos, np.add(fp_count, tn_count)))
        precision_vec = np.divide(tp_count, predicted_pos)
        recall_vec = np.divide(tp_count, true_pos)
        f1_vec = np.divide(np.multiply(2.0,np.multiply(precision_vec, recall_vec)), np.add(precision_vec, recall_vec))
        print("Accuracy:")
        print(accuracy_vec)
        print("Precision:")
        print(precision_vec)
        print("Recall:")
        print(recall_vec)
        print("F1:")
        print(f1_vec)


def evaluate(model_spec, model_dir, params, restore_from, print_all=False, find_confusion=False, find_bad_images=False, find_metrics=True, find_perclass_metrics=False):
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
            metrics = evaluate_sess(sess, model_spec, num_steps, print_all=print_all, find_confusion=find_confusion, find_bad_images=find_bad_images, find_metrics=find_metrics, find_perclass_metrics=find_perclass_metrics)
            metrics_name = '_'.join(restore_from.split('/'))
            save_path = os.path.join(model_dir, "metrics_test_{}.json".format(metrics_name))
            save_dict_to_json(metrics, save_path)
        else:
            evaluate_sess(
                sess,
                model_spec,
                num_steps,
                print_all=print_all,
                find_confusion=find_confusion,
                find_bad_images=find_bad_images,
                find_metrics=find_metrics,
                find_perclass_metrics=find_perclass_metrics
        )
