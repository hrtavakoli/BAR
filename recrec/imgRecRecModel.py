
#
#   This is a simple recurrent recognition model
#


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from recrec.opslib import encode_image
from recrec.opslib import imgprocess
from recrec.opslib import inputdata


class ImgRecRecModel(object):
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

        # endpoint variables
        self.end_points = None

        # global step Tensor
        self.global_step = None

        # handler to the initialization function
        self.init_fn = None

        #self.num_lstm_units = config.num_classes
        # what if we keep this number fixed to 10
        self.num_lstm_units = 10

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
            image_feed = tf.placeholder(dtype=tf.uint8, shape=[self.config.num_sequence, 224, 224, 3], name="image_feed")
            # Process image and insert batch dimensions.

            image_feed = tf.unstack(image_feed)
            pre_processed_images = []
            for c_img in image_feed:
                proc_image = tf.expand_dims(imgprocess.process_image(c_img, is_training=self.is_training(),
                                                                     height=self.config.image_height,
                                                                     width=self.config.image_width), 0)
                pre_processed_images.append(proc_image)
            images = tf.stack(pre_processed_images)

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

                serialized_queue = input_queue.dequeue()

                [images, targets] = inputdata.parse_sequence_data_example(serialized_queue,
                                                                          colorChan=self.config.color,
                                                                          patchSize=self.config.image_height,
                                                                          seq_len=self.config.num_sequence,
                                                                          one_hot=False,
                                                                          n_labels=self.config.num_classes)

                #images = tf.Print(images, [images.shape])

                # uncomment for the Poet
                #images = tf.reshape(images, [self.config.image_width,
                #                             self.config.image_height,
                #                             self.config.color,
                #                             self.config.num_sequence])
                #images = tf.transpose(images, [3, 0, 1, 2])
                # the dimnesions for ImageNet
                images = tf.reshape(images, [self.config.num_sequence,
                                             self.config.image_width,
                                             self.config.image_height,
                                             self.config.color])


                images = tf.unstack(images)
                pre_processed_images = []
                for c_im in images:
                    pre_processed_image = imgprocess.process_image(image=c_im,
                                                                    is_training=self.is_training(),
                                                                    height=self.config.image_height,
                                                                    width=self.config.image_width)
                    pre_processed_images.append(pre_processed_image)
                pre_processed_images = tf.stack(pre_processed_images)
                image_and_label.append([pre_processed_images, targets])

            # Create batch of inputs

            images, targets = inputdata.fetch_dynamic_batch_of_data(decoded_data=image_and_label,
                                                                    batch_size=self.config.batch_size,
                                                                    num_threads=self.config.num_preprocess_threads,
                                                                    min_after_dequeue=100)
            #images = tf.Print(images, [images, targets])
        self.images = images
        self.targets = targets

    def build_model(self):
        """ builds the graph of the model

        :return:
        """
        # Start building the recurrent part of the network.
        if self.mode == "inference":
            images = self.images
        else:
            images = tf.transpose(self.images, [1, 0, 2, 3, 4])

        images = tf.unstack(images)
        output = []


        with tf.variable_scope("lstm") as lstm_scope:
            #lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.num_lstm_units,
            #                                         state_is_tuple=True)
            lstm_cell = tf.contrib.rnn.GRUCell(num_units=self.num_lstm_units)
            if self.is_training():
                lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell,
                                                          input_keep_prob=self.config.dropout_rate,
                                                          output_keep_prob=self.config.dropout_rate)
            # Initialize the zero state.
            if self.mode == "inference":
                _state = lstm_cell.zero_state(batch_size=1, dtype=tf.float32)
            else:
                _state = lstm_cell.zero_state(batch_size=self.config.batch_size, dtype=tf.float32)

            for image in images:

                if self.config.CNN_encoder == "ResNet50":
                    net, ep = encode_image.res50_encode(image,
                                                        trainable=self.config.fine_tune_cnn,
                                                        is_training=self.is_training())
                    self.end_points = tf.get_collection(tf.GraphKeys.GLOBAL_STEP, scope="lstm/resnet_v2_50")
                else:
                    if self.config.CNN_encoder == "VGG":
                        net, ep = encode_image.vgg_encode(image,
                                                          trainable=self.config.fine_tune_cnn,
                                                          is_training=self.is_training(),
                                                          dropout_keep_prob=self.config.dropout_rate)
                        #self.end_points = ep
                        self.end_points = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="lstm/vgg_16")
                    else:
                        if self.config.CNN_encoder == "Alex":
                            net, ep = encode_image.alexnet_encode2(image,
                                                                   num_classes=self.num_lstm_units,
                                                                  #trainable=self.config.trainable,
                                                                   is_training=self.is_training(),
                                                                   dropout_keep_prob=self.config.dropout_rate)
                            self.end_points = ep
                        else:
                            net, ep = encode_image.hdnn_encode(image,
                                                               num_classes=self.num_lstm_units,
                                                               #trainable=self.config.trainable,
                                                               is_training=self.is_training(),
                                                               dropout_keep_prob=self.config.dropout_rate)
                            self.end_points = ep

                lstm_output, _state = lstm_cell(inputs=net, state=_state)
                lstm_scope.reuse_variables()
                #tf.concat(axis=1, values=_state, name="state")
                #output.append(lstm_output)

        #output = tf.stack(output)
        tf.summary.scalar("LSTM/state", _state)
        output = _state
        output = tf.transpose(output, [1, 0, 2])
        if self.mode == "inference":
            output = tf.reshape(output, [1, -1])
        else:
            output = tf.reshape(output, [self.config.batch_size, -1])

        with tf.variable_scope("logits") as logits_scope:
            logits = tf.contrib.layers.fully_connected(
                inputs=output,
                num_outputs=self.config.num_classes,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                scope=logits_scope)

        if self.mode == "inference":

            tf.nn.softmax(logits, name="softmax")
            print("inference")

        else:

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=logits)
            batch_loss = tf.reduce_mean(losses, name="batch_loss")
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
            print("Restoring CNN variables from checkpoint file {}",
                                self.config.CNN_ckpt)
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
        #with tf.device('/cpu:0'):
        self.build_inputs()

        self.build_model()

        if self.config.CNN_encoder == "VGG":
            self.setup_cnn_initializer()

        self.setup_global_step()
