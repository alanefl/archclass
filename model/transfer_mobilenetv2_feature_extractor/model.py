"""Defines a model that uses the last layer of MobileNet V2
as a feature extractor into a linear softmax classifier.

See https://tfhub.dev/google/imagenet/mobilenet_v2_035_128/feature_vector/2.

"""
import tensorflow as tf
import tensorflow_hub as hub


def get_mobilenetv2_feature_extractor_module():
    #return hub.Module("https://tfhub.dev/google/imagenet/mobilenet_v2_035_128/feature_vector/2")
    return hub.Module("https://drive.google.com/uc?export=view&id=1sXomHtEgTqfSwuZjwrctsUgUNU9mkRFy")

def build_transfer_mobilenetv2_feature_extractor_model(inputs, params, reuse=False, is_training=False):
    """Using pretrained MobileNet V2 as a feature extractor before a softmax classifier we train.
    :return: Logits or output distribution of the model.
    """
    with tf.variable_scope('transfer_mobilenetv2_feature_extractor_model', reuse=reuse):
        module = get_mobilenetv2_feature_extractor_module()
        height, width = hub.get_expected_image_size(module)
        print(height, width)
        exit(0)

        img_size = params.image_size
        images = inputs['images']
        num_labels = params.num_labels
        assert images.get_shape().as_list() == [None, img_size, img_size, 3]
        return tf.layers.dense(tf.layers.flatten(images), num_labels)
