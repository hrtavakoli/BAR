#####
# Script to read the input data into the system
#####

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def distort_image(image, rid):
    """get an image and perturb the values.
         
    :param image: a float32 image which is of size [height width 3] with values between [0 1]
    :param rid: is a random id to decide how one should be handle the color channel 
    :return: 
        distorted_image: A float32 Tensor of shape [height width 3] with values in range [-1 1]
    """

    # randomly flip the image
    image = tf.image.random_flip_left_right(image)

    color_ordering = rid % 2
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.032)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    else:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.032)

    image = tf.clip_by_value(image, 0.0, 1.0)

    return image


def process_image(image, is_training, height, width,
                  resize_height=240, resize_width=240, rid=0):
    """
        gets an image, resize and apply random distortions for training
    :param image: a float32 image which is of size [height width 3] with values uint8
    :param is_training: if it is a training sequence
    :param height: height of the image
    :param width: width of the image
    :param resize_height: the resize size of the height
    :param resize_width: the resize size of the width
    :param rid: it is a randomness factor id used in the distortion process of the images to decide which distortion one shoudl apply 
    :return  
            returns a distorted image with values between [0 1]
    """

    # Helper function to log an image summary to the visualizer. Summaries are
    # only logged in thread 0.
    def image_summary(name, image):
        if not rid:
            tf.summary.image(name, tf.expand_dims(image, 0))

    # convert the image to a float32 with values between [0 1]
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image_summary("original_image", image)

    # resize the image
    assert (resize_height > 0) == (resize_width > 0)
    if resize_height:
        image = tf.image.resize_images(image, size=[resize_height, resize_width],
                                       method=tf.image.ResizeMethod.BILINEAR)
    # Crop to the useful size
    if is_training:
        image = tf.random_crop(image, [height, width, 3])
    else:
        image = tf.image.resize_image_with_crop_or_pad(image, height, width)

    image_summary("resized_image", image)

    if is_training:
        image = distort_image(image, rid)

    image_summary("final_image", image)

    # scale the image to the range of [-1 1]
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image

