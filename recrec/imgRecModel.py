

#
#   This is a simple image recognition model
#   It is designed to play the role of the baseline for a system
#


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from recrec.opslib import encode_image
from recrec.opslib import imgprocess
from recrec.opslib import inputdata

slim = tf.contrib.slim

class ImgRecModel(object):
    """
        This is a simple class to perform image recognition in its traditional way
    """

    def __init__(self, config, mode):
        """
        Initialization

        :param config:  configuration parameters
        :param mode:    "train" or "inference"

        """
        assert mode in ["train", "inference"]
        self.config = config
        self.mode = mode

        # set up a reader
        self.reader = tf.TFRecordReader()

        # A tensor to keep track of images [batch, height, width, channel]
        self.images = None

        # A tensor to keep the labels of the images [batch, label]
        self.targets = None

        # A scalar value to keep the total loss value
        self.total_loss = None


        # global step Tensor
        self.global_step = None

    def is_training(self):
        """
        :return: if the model is in trainign mode
        """
        return self.mode == "train"

    def build_inputs(self):
        """
            builds the input variables of the session graph
        :return:
            self.images
            self.targets
        """
        if self.mode == "inference":
            image_feed = tf.placeholder(dtype=tf.uint8, shape=[224, 224, 3], name="image_feed")
            # Process image and insert batch dimensions.
            images = tf.expand_dims(imgprocess.process_image(image_feed, is_training=self.is_training(),
                                                             height=self.config.image_height,
                                                             width=self.config.image_width), 0)
            # There is not target
            targets = None
        else:
            # Prefetch images for training

            input_queue = inputdata.prefetch_input_data(self.reader,
                                                        self.config.input_file_pattern,
                                                        is_training=self.is_training(),
                                                        batch_size=self.config.batch_size,
                                                        values_per_shard=self.config.values_per_input_shard,
                                                        input_queue_capacity_factor=self.config.input_queue_capacity_factor,
                                                        num_reader_threads=self.config.num_input_reader_threads)

            # fetch the images, distort and feed
            # create multiple thrids and distort image in regard to the thrid id
            #assert self.config.num_preprocess_threads % 2 == 0
            image_and_label = []
            for thrid_id in range(self.config.num_preprocess_threads):

                serialzied_queue = input_queue.dequeue()

                [images, targets] = inputdata.parse_image_data_example(serialzied_queue,
                                                                       n_labels=self.config.num_classes,
                                                                       colorChan=self.config.color,
                                                                       #patchSize=self.config.image_height,
                                                                       one_hot=False)

                images = tf.reshape(images, [self.config.image_width,
                                             self.config.image_height,
                                             self.config.color])

                pre_processed_images = imgprocess.process_image(image=images, is_training=self.is_training(),
                                                  height=self.config.image_height,
                                                  width=self.config.image_width)


                image_and_label.append([pre_processed_images, targets])

            # Create batch of inputs

            images, targets = inputdata.fetch_dynamic_batch_of_data(decoded_data=image_and_label,
                                                                    batch_size=self.config.batch_size,
                                                                    num_threads=self.config.num_preprocess_threads,
                                                                    min_after_dequeue=100)

        self.images = images
        self.targets = targets


    def build_model(self):

        if self.config.CNN_encoder == "VGG":
            with tf.variable_scope('VGG', 'VGG', [self.images]) as sc:
                end_points_collection = sc.name + '_end_points'

                net, ep = encode_image.vgg_encode(self.images,
                                                  trainable=self.config.fine_tune_cnn,
                                                  is_training=self.is_training(),
                                                  dropout_keep_prob=self.config.dropout_rate)
                with slim.arg_scope([slim.fully_connected],
                                    outputs_collections=[end_points_collection], trainable=True):
                    net = slim.fully_connected(net, self.config.num_classes, activation_fn=None, scope='myfunc')
        else:
            if self.config.CNN_encoder == "Alex":
                net, ep = encode_image.alexnet_encode2(self.images,
                                                       num_classes=self.config.num_classes,
                                                       trainable=True,
                                                       is_training=self.is_training(),
                                                       dropout_keep_prob=self.config.dropout_rate)
                self.end_points = ep
            else:
                net, ep = encode_image.hdnn_encode(self.images,
                                                   num_classes=self.config.num_classes,
                                                   trainable=True,
                                                   is_training=self.is_training(),
                                                   dropout_keep_prob=self.config.dropout_rate)
                self.end_points = ep

        self.end_points = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="VGG/vgg_16")

        for v in self.end_points:
            tf.contrib.layers.summaries.summarize_activation(v)
        """net, ep = encode_image.alexnet_encode(self.images,
                                              trainable=True,
                                              is_training=self.is_training(),
                                              dropout_keep_prob=self.config.dropout_rate)
                                              """

        if self.mode == "inference":
            tf.nn.softmax(net, name="softmax")
            print("inference")
        else:
            # compute the loss
            #self.targets = tf.Print(self.targets, [self.targets])
            #net = tf.Print(net, [net[0], tf.shape(net)])
            #net = tf.Print(net, [net])
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=net)

            batch_loss = tf.reduce_mean(losses, name="batch_loss")
            #batch_loss = tf.Print(batch_loss, [batch_loss])
            tf.losses.add_loss(batch_loss)
            total_loss = tf.losses.get_total_loss()

            # add summarize

            tf.summary.scalar("losses/batch_loss", batch_loss)
            tf.summary.scalar("losses/total_loss", total_loss)
            for var in tf.trainable_variables():
                tf.summary.histogram("parameters/" + var.op.name, var)

            self.total_loss = total_loss

    def setup_cnn_initializer(self):
        """Sets up the function to restore inception variables from checkpoint."""
        if self.mode != "inference":
            # Restore inception variables only.
            saver = tf.train.Saver(self.end_points)

            def restore_fn(sess):
                tf.logging.info("Restoring CNN variables from checkpoint file %s",
                                self.config.CNN_ckpt)
                saver.restore(sess, self.config.inception_checkpoint_file)

            self.init_fn = restore_fn

    def setup_global_step(self):
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
        self.global_step = global_step

    def build(self):
        """Creates all ops for training and evaluation."""
        self.build_inputs()
        self.build_model()

        if self.config.CNN_encoder == "VGG":
            print("loading vgg model")
            self.setup_cnn_initializer()

        self.setup_global_step()
