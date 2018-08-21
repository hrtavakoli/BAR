
#
#   file to encode the images with a CNN
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets.vgg import vgg_16
from tensorflow.contrib.slim.python.slim.nets.resnet_v2 import resnet_v2_50


slim = tf.contrib.slim


def res50_encode(inputs,
                 trainable=False,
                 is_training=False,
                 add_summaries=True):
    fine_tune = is_training & trainable
    net, end_points = resnet_v2_50(inputs,
                                   is_training=fine_tune,
                                   scope="resnet_v2_50")

    net = tf.squeeze(net, [1, 2], name='resnet_v2_50/squeezed')
    if add_summaries:
        for v in end_points.values():
            tf.contrib.layers.summaries.summarize_activation(v)

    return net, end_points



def vgg_encode(inputs,
               trainable=False,
               is_training=False,
               dropout_keep_prob=0.8,
               add_summaries=True):

    fine_tune = is_training & trainable
    net, end_points = vgg_16(inputs,
                             is_training=fine_tune,
                             dropout_keep_prob=dropout_keep_prob,
                             spatial_squeeze=True,
                             scope='vgg_16')
  # Add summaries
    if add_summaries:
        for v in end_points.values():
            tf.contrib.layers.summaries.summarize_activation(v)

    return net, end_points


def hdnn_encode(inputs,
                num_classes,
                trainable=True,
                is_training=True,
                dropout_keep_prob=0.8,
                add_summaries=True,
                scope="HDNNet"):
    """ function to encode an input using my net architecture """

    # this is useful for training the model or fine-tuning it

    is_model_training = trainable and is_training
    with tf.variable_scope(scope, 'HDNNet', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'

        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                trainable=trainable):
                net = slim.conv2d(inputs, 64, [11, 11], stride=4, padding='SAME', scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
                net = slim.conv2d(net, 128, [5, 5], padding='SAME', scope='conv2')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool2')
                net = slim.conv2d(net, 256, [3, 3], stride=2, padding='SAME', scope='conv3')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool3')

                net = slim.flatten(net, scope='flatten')

                net = slim.fully_connected(net, 256, scope='fc6')

                net = slim.dropout(net,keep_prob=dropout_keep_prob,
                                       is_training=is_model_training,
                                       scope="dropout6")
                net = slim.fully_connected(net, 128, scope='fc7')

                net = slim.dropout(net,
                                   keep_prob=dropout_keep_prob,
                                   is_training=is_model_training,
                                   scope="dropout7")

                net = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc8')

        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        #net = tf.Print(net, [net])
        #net = tf.squeeze(net, name='fc8/squeezed')
        #end_points[sc.name + '/fc8'] = net

    # Add summaries
    if add_summaries:
        for v in end_points.values():
            tf.contrib.layers.summaries.summarize_activation(v)
    return net, end_points

def alexnet_encode2(inputs,
                    num_classes,
                    trainable=True,
                    is_training=True,
                    dropout_keep_prob=0.8,
                    add_summaries=True,
                    scope="AlexNet"):
    """ function to encode an input using my net architecture """

    # this is useful for training the model or fine-tuning it

    is_model_training = trainable and is_training
    with tf.variable_scope(scope, 'AlexNet', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'

        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                trainable=trainable):
                net = slim.conv2d(inputs, 64, [11, 11], 1, padding='SAME', scope='conv1')

                net = slim.conv2d(net, 64, [5, 5], 1, padding='SAME', scope='conv2')
                net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')

                net = slim.conv2d(net, 128, [3, 3], 1, padding='SAME', scope='conv3')
                net = slim.max_pool2d(net, [3, 3], 2, scope='pool3')

                net = slim.flatten(net, scope='flatten')

                net = slim.fully_connected(net, 256, scope='fc6')

                net = slim.dropout(net,keep_prob=dropout_keep_prob,
                                       is_training=is_model_training,
                                       scope="dropout6")
                net = slim.fully_connected(net, 128, scope='fc7')

                net = slim.dropout(net,
                                   keep_prob=dropout_keep_prob,
                                   is_training=is_model_training,
                                   scope="dropout7")

                net = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc8')

        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        #net = tf.Print(net, [net])
        #net = tf.squeeze(net, name='fc8/squeezed')
        #end_points[sc.name + '/fc8'] = net

    # Add summaries
    if add_summaries:
        for v in end_points.values():
            tf.contrib.layers.summaries.summarize_activation(v)
    return net, end_points

def alexnet_encode(inputs,
                   trainable=True,
                   is_training=True,
                   dropout_keep_prob=0.8,
                   add_summaries=True,
                   scope="AlexNet"):
    """ function to encode an input using my net architecture """

    # this is useful for training the model or fine-tuning it

    is_model_training = trainable and is_training
    with tf.variable_scope(scope, 'AlexNet', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'

        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                trainable=trainable):
                net = slim.conv2d(inputs, 64, [11, 11], 1, padding='SAME', scope='conv1')

                net = slim.conv2d(net, 64, [5, 5], 1, padding='SAME', scope='conv2')
                net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')

                net = slim.conv2d(net, 128, [3, 3], 1, padding='SAME', scope='conv3')
                net = slim.max_pool2d(net, [3, 3], 2, scope='pool3')

                net = slim.flatten(net, scope='flatten')

                net = slim.fully_connected(net, 256, scope='fc6')

                net = slim.dropout(net,keep_prob=dropout_keep_prob,
                                       is_training=is_model_training,
                                       scope="dropout6")
                net = slim.fully_connected(net, 128, scope='fc7')

                net = slim.dropout(net,
                                   keep_prob=dropout_keep_prob,
                                   is_training=is_model_training,
                                   scope="dropout7")

                #net = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc8')

        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        #net = tf.Print(net, [net])
        #net = tf.squeeze(net, name='fc8/squeezed')
        #end_points[sc.name + '/fc8'] = net

    # Add summaries
    if add_summaries:
        for v in end_points.values():
            tf.contrib.layers.summaries.summarize_activation(v)
    return net, end_points


