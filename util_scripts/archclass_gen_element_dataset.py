"""
Generates the architecture elem parts dataset.

Must be run within TensorFlow's model repo at the path:

    models/research/object_detection/

Runs a pretrained object detector on all JPG images given under specified dev, train, and test folders.

E.g.:

    Given: 

        train/class1-0.jpg
        train/class1-1.jpg
        train/class2-0.jpg

        dev/class1-1.jpg
        dev/class1-2.jpg
        dev/class2-1.jpg

        test/class1-2.jpg
        test/class1-3.jpg
        test/class2-2.jpg

    Will generate outputs:

        train/elements/elem-class1-0-[elem label]-[idx].jpg
        train/elements/elem-class1-1-[elem label]-[idx].jpg
        train/elements/elem-class2-0-[elem label]-[idx].jpg

        etc.

    Where 'elem label' is the label of the architectural element extracted from
    the corresponding full image, and [idx] is the index of that element class in the
    given example.

Adapted from the iPython notebook provided by Google's Object Detection API.

"""

import argparse
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from tqdm import tqdm

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

from utils import label_map_util
from utils import visualization_utils as vis_util

# The model we expect to find in the current directory.
MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'oid_bbox_trainable_label_map.pbtxt')

# Score threshold above which to keep object bounding boxes
OBJ_DETECTION_THRESHOLD = 0.5

# Maps numerical indices to the corresponding object class
CATEGORY_INDEX = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# The IDs and classes we care about for the architectural classification task.
ARCH_CLASSES = [
    (165, u'Bronze sculpture'),
    (9, u'Building'),
    (127, u'Castle'),
    (187, u'Clock'),
    (58, u'Door'),
    (85, u'Flag'),
    (20, u'House'),
    (160, u'Lighthouse'),
    (49, u'Sculpture'),
    (46, u'Skyscraper'),
    (69, u'Street light'),
    (116, u'Stairs'),
    (44, u'Tower'),
    (19, u'Window'),
]
ARCH_CLASS_IDS = [x[0] for x in ARCH_CLASSES]
ARCH_CLASS_NAMES = [x[1] for x in ARCH_CLASSES]

parser = argparse.ArgumentParser()
parser.add_argument('--src_data_dir', required=True,
                    help="Directory containing the dev, train, and test directories with images \
                    -- also where the detection output will go (dev/elements, train/elements,\
                    test/elements)"
)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)
    ).astype(np.uint8)

def run_inference_on_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

def detect_single_image(image_np, graph):
    """
    Runs the object detector on a single image given as a numpy array of 
    dimensions (height, width, 3).  Returns boxes, classes, and scores, 
    np arrays of the same length corresponding to the bounding box, class,
    and score of the detected objects.
    """
    output_dict = run_inference_on_single_image(image_np, graph)
    detection_boxes = output_dict['detection_boxes']
    detection_classes = output_dict['detection_classes']
    detection_scores = output_dict['detection_scores']
    detection_classes = [CATEGORY_INDEX[class_id]['name'] for class_id in detection_classes]
    return (detection_boxes, detection_classes, detection_scores)

def save_detected_objects(boxes, classes, scores, image_np, image_name, destdir):
    """
    Given the outputs of `detect_single_image`, the np image itself,
    the original filename, and a destination directory, selects the boxes and classes
    that pass a certain criteria for our purposes and saves the image elements as
    images in the given directory.
    """
    print("Saving objects detected for image %s in directory %s." % (image_name, destdir))
    arch_style, sample = image_name.split("-")
    sample = sample.split(".")[0]

    elem_class_counts = {}
    for box, obj_class, score in zip(boxes, classes, scores):
        if score <= OBJ_DETECTION_THRESHOLD:
            continue
        if obj_class not in ARCH_CLASS_NAMES:
            continue
        
        print(box, obj_class, score)

        if obj_class not in elem_class_counts:
            elem_class_counts[obj_class] = 0

        elem_class_counts[obj_class] += 1
        elem_file_name = "elem-%s-%s-%s-%d.jpg" % (arch_style, sample, obj_class, elem_class_counts[obj_class])
        elem_file_path = os.path.join(destdir, elem_file_name)
        print("Saving image from source %s to %s" % (image_name, elem_file_path))

        to_crop = np.array(image_np)
        height, width, _ = to_crop.shape
        ymin, xmin, ymax, xmax = box
        ymin = int(height * ymin)
        xmin = int(width * xmin)
        ymax = int(height * ymax)
        xmax = int(width * xmax)
        crop = np.array(to_crop)[ymin:ymax, xmin:xmax, :]
        im = Image.fromarray(crop)
        im.save(elem_file_path)


def extract_objects(filepaths, destdir, graph):
    """
    Run pretrained object detector on each of the files in `filepaths.` Outputs to `destdir`.
    The `stage` is dev, train, or test.
    """
    print("Extracting objects to directory %s" % (destdir))
    for filepath in tqdm(filepaths):
        try:
            image = Image.open(filepath)
            image_np = load_image_into_numpy_array(image)
            boxes, classes, scores = detect_single_image(image_np, graph)
            save_detected_objects(boxes, classes, scores, image_np, os.path.basename(filepath), destdir)
        except Exception as e:
            print("An exception occurred")
            print("Skipping file: %s" % filepath)
            print(e)

def main(args):

    # Load the frozen TensorFlow model into Memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    path_to_train_images = os.path.join(args.src_data_dir, 'train')
    path_to_dev_images = os.path.join(args.src_data_dir, 'dev')
    path_to_test_images = os.path.join(args.src_data_dir, 'test')
    
    paths = [('train', path_to_train_images), ('dev', path_to_dev_images), ('test', path_to_test_images)]

    for stage, dir_path in paths:
        ct = 0
        images_to_process = []
        for filename in os.listdir(dir_path):
            if filename.endswith(".jpg"): 
                images_to_process.append(os.path.join(dir_path, filename))
                ct += 1
        print("Processing %d %s set images" % (ct, stage))

        dest_dir = os.path.join(dir_path, 'elements')
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        else:
            print("Warning: dataset output dir {} already exists".format(dest_dir))

        extract_objects(images_to_process, dest_dir, detection_graph)
        print("Done processing %d %s set images\n" % (ct, stage))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
