#
# Get a sequence of images and compute the CNN results on it
#
#


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from imgprocess import process_image


def build_sequence_input(sequence, label, is_training, colorChan=3, patchSize=224, seq_len=3):
    """
        Gets one image sequence and make it ready to be processed
    
    :param sequence: a sequence of images
    :param label: the label of the sequence
    :param is_training: 
    :param colorChan: the number of color channels in the sequence
    :param patchSize: the size of the image (we call it patch size)
    :param seq_len: the length of the sequence
    :return: 
    """

    sequence = tf.reshape(sequence, [seq_len, patchSize, patchSize, colorChan])

    # unpack the sequence
    sequence = tf.unstack(sequence)
    new_sequence = []
    for s in sequence:
        img = process_image(s, is_training, patchSize, patchSize)
        new_sequence.append(img)

    sequence = tf.stack(new_sequence)

    return sequence, label




