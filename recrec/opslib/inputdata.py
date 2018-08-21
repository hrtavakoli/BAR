#####
# Script to read the input data into the system
#####

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf



def parse_image_data_example(serialized, n_labels, colorChan=3, patchSize=224, one_hot=False):
    """ parses a tensorflow.sequenceExample into the data structure of patches and labels and ensure the length of sequence is 10
        Args:
            :param serialized: A scalar string Tensor; a single serialized Sequence Example
            :param colorChan: The expected size of color channel in images default is 3 for color images
            :param patchSize: The size of an image patch is patchSizeXpatchSize, the defualt is 224 for AlexNet
            :param one_hot: to retrieve the label as a one hot vector
            :param n_labels: is the total number of labels

        Returns:
            :return image_sequence: A scalar tensor containing an image sequence
            :return  label: of the image as a one-hot vector or a number
    """
    features = tf.parse_single_example(serialized,
                                       features={
                                           'label': tf.FixedLenFeature([], dtype=tf.int64),
                                           'image': tf.FixedLenFeature([], dtype=tf.string)
                                       }
                                       )
    if one_hot:
        # we assume labels start from 1 so we adjust to make it suitable for one_hot function
        label = tf.one_hot(features['label'], depth=n_labels, on_value=1, off_value=0)
    else:
        label = features['label']

    imageData = tf.decode_raw(features['image'], tf.uint8)
    image_size = colorChan * patchSize * patchSize
    imageData.set_shape([image_size])
    #imageData.set_shape([patchSize, patchSize, colorChan])
    #label = tf.Print(label, [label])

    return imageData, label

def parse_sequence_data_example(serialized, colorChan=3, patchSize=224, seq_len=3, one_hot=False, n_labels=3):
    """ parses a tensorflow.sequenceExample into the data structure of patches and labels and ensure the length of sequence is 10
        Args:
            :param serialized: A scalar string Tensor; a single serialized Sequence Example
            :param colorChan: The expected size of color channel in images default is 3 for color images
            :param patchSize: The size of an image patch is patchSizeXpatchSize, the defualt is 299 for google networks
            :param seq_len: is the length of the sequential data, default is 10
            :param one_hot: to retrieve the label as a one hot vector
            :param n_labels: is the total number of labels

        Returns:
            :return image_sequence: A scalar tensor containing an image sequence                        
            :return  label: of the image as a one-hot vector or a number 
    """
    features = tf.parse_single_example(serialized,
                                       features={
                                           'label': tf.FixedLenFeature([], dtype=tf.int64),
                                           'patches': tf.FixedLenFeature([], dtype=tf.string)
                                       }
                                       )
    if one_hot:
        # we assume labels start from 1 so we adjust to make it suitable for one_hot function
        label = tf.one_hot(features['label'], depth=n_labels, on_value=1, off_value=0)
    else:
        label = features['label']
#        label = tf.Print(label, [label])

    #label.set_shape([1])
    image_sequence = tf.decode_raw(features['patches'], tf.uint8)
    # label = tf.reshape(label, tf.stack([n_label]))
    # image_sequence = tf.reshape(image_sequence, tf.stack([n_sequence, colorChan, patchSize, patchSize]))
    sequence_len = (seq_len) * colorChan * patchSize * patchSize
    image_sequence.set_shape([sequence_len])

    # we ensure the image sequence has maximum length

    # image_sequence.set_shape([expected_len])

    return image_sequence, label


def fetch_dynamic_batch_of_data(decoded_data, batch_size=30, num_threads=1, min_after_dequeue=5):
    """
    This is a function to fetch dynamic size batch of data for image decoding 
     Since we have a dynamic sequences, if the length of sequence is smaller than max_len, it will be padded by zero
    
    :param decoded_data: is the decoded image data consisting of [image_sequence, label]
    :param batch_size: is the size of the batch size
    :param num_threads: is the number of threads, equal to 1 
    :param min_after_dequeue:    
    :return: 
    """
    # There is no such a function to handle sequence of images with different length.
    # We here try to make the images with the same sequence length

    # First pre-process the data for the batching ensure everything has the max_len

    # min_after_dequeue + (num_threads + a small safety margin) * batch_size

    capacity = min_after_dequeue + (num_threads + 20) * batch_size

    data_batch, labels = tf.train.batch_join(decoded_data,
                                             batch_size=batch_size,
                                             capacity=capacity,
                                             allow_smaller_final_batch=True)
    return data_batch, labels


def prefetch_input_data(reader,
                        file_pattern,
                        is_training,
                        batch_size,
                        values_per_shard,
                        input_queue_capacity_factor=16,
                        num_reader_threads=1,
                        shard_queue_name="filename_queue",
                        value_queue_name="input_queue"):
    """
    Prefetches string values from disk into an input queue.

    In training the capacity of the queue is important because a larger queue
    means better mixing of training examples between shards. The minimum number of
    values kept in the queue is values_per_shard * input_queue_capacity_factor,
    where input_queue_memory factor should be chosen to trade-off better mixing
    with memory usage.

    :param reader: Instance of tf.reader
    :param file_pattern: A comma seperated list of file patterns (e.g. ./data/train_image)
    :param is_training: the training
    :param batch_size: The batch size of the model to determine queue capacity
    :param values_per_shard: Approximate number of values per shard
    :param input_queue_capacity_factor: Minimum number of values to keep in the queue
      in multiples of values_per_shard. See comments above.
    :param num_reader_threads: Number of reader threads to fill the queue
    :param shard_queue_name: Name for the shards filename queue
    :param value_queue_name: Name for the values input queue
    :return:
            return a queue
    """

    data_files = []
    for pattern in file_pattern.split(","):
        data_files.extend(tf.gfile.Glob(pattern))
        if not data_files:
            tf.logging.fatal("Found no input files matching %s", file_pattern)
        else:
            tf.logging.info("Prefetching values from %d files matching %s", len(data_files), file_pattern)

    if is_training:
        filename_queue = tf.train.string_input_producer(data_files, shuffle=True, capacity=16, name=shard_queue_name)
        min_queue_examples = values_per_shard * input_queue_capacity_factor
        capacity = min_queue_examples + 100 * batch_size
        values_queue = tf.RandomShuffleQueue(capacity=capacity,
                                             min_after_dequeue=min_queue_examples,
                                             dtypes=[tf.string],
                                             name="random_" + value_queue_name)
    else:
        filename_queue = tf.train.string_input_producer(data_files, shuffle=False, capacity=1, name=shard_queue_name)
        capacity = values_per_shard + 3 * batch_size
        values_queue = tf.FIFOQueue(capacity=capacity, dtypes=[tf.string], name="fifo_" + value_queue_name)

    enqueue_ops = []
    for _ in range(num_reader_threads):
        _, value = reader.read(filename_queue)
        enqueue_ops.append(values_queue.enqueue([value]))
    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(values_queue, enqueue_ops))
    tf.summary.scalar("queue/%s/fraction_of_%d_full" % (values_queue.name, capacity),
                      tf.cast(values_queue.size(), tf.float32) * (1. / capacity))

    return values_queue

