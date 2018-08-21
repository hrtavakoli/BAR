from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import os
import cv2

import matplotlib.pyplot as plt



from recrec import imgRecRecModel
from recrec import configuration

FLAGS = tf.app.flags.FLAGS

#tf.flags.DEFINE_string("input_file_directory", "/home/rtavah1/projects/affect2/data/testPatches/",
#                       "File pattern of input files.")

tf.flags.DEFINE_string("input_file_directory", "/home/rtavah1/projects/databases/imgNet/val/",
                       "File pattern of input files.")
#tf.flags.DEFINE_string("chk_model", "./data/train/imgrec/model.ckpt-493156",
tf.flags.DEFINE_string("chk_model", "/home/rtavah1/projects/poet/log_full2/model.ckpt-492356",
#tf.flags.DEFINE_string("chk_model", "./data/recrec_model/model.ckpt-60000",
                       "the checkpoint to load model")
tf.flags.DEFINE_boolean("crop_only", False,
                        "We use only crops.")


tf.logging.set_verbosity(tf.logging.INFO)


def restore_model(checkpoint_path, saver):
    """
    Creates a function that restores a model from checkpoint.

    :param checkpoint_path: Checkpoint file or a directory containing a checkpoint
        file.
    :param saver: Saver for restoring variables from the checkpoint file.
    :return:
        restore_fn: A function such that restore_fn(sess) loads model variables
        from the checkpoint file.
    :raises
      ValueError: If checkpoint_path does not refer to a checkpoint file or a
        directory containing a checkpoint file.


    """
    if tf.gfile.IsDirectory(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        if not checkpoint_path:
            raise ValueError("No checkpoint file found in: %s" % checkpoint_path)

    def _restore_fn(sess):
        tf.logging.info("Loading model from checkpoint: %s", checkpoint_path)
        saver.restore(sess, checkpoint_path)
        tf.logging.info("Successfully loaded checkpoint: %s",
                        os.path.basename(checkpoint_path))

    return _restore_fn


def build_graph_from_checkpoint(checkpoint_path):
     """
     Load the model from a graph
     :param checkpoint_path: checkpoint file or a directory containing a checkpoint
     :return restore_fn: A function such that restore_fn(sess) loads model variables from the checkpoint file.

     """
     tf.logging.info("Building model.")
     saver = tf.train.Saver()
     return restore_model(checkpoint_path, saver)


def main(unused_argv):

    model_config = configuration.CNNModelConfig()
    model_config.input_file_pattern = FLAGS.input_file_directory

    model_config.CNN_encoder = "HDNet"
    model_config.fine_tune_cnn = False
    model_config.num_classes = 101
    model_config.values_per_input_shard = 150
    model_config.batch_size = 1
    model_config.num_sequence = 3
    # Load the files in the input file directory
    dir_list = os.listdir(FLAGS.input_file_directory)

    # Build the inference graph.
    g = tf.Graph()
    with g.as_default():
        # build the model for inference
        model = imgRecRecModel.ImgRecRecModel(model_config,
                                              mode="inference")
        model.build_inputs()
        model.build_model()
        restore_function = build_graph_from_checkpoint(FLAGS.chk_model)
    g.finalize()

    res_file = open("./results/result_HDNet_imgnet_nc3_492k.txt", "w")

    # Start a session and start guessing image type
    with tf.Session(graph=g) as sess:
        restore_function(sess)

        for dirname in dir_list:
            img = []

            dir_path = os.path.join(FLAGS.input_file_directory, dirname)
            files = os.listdir(dir_path)
            if FLAGS.crop_only:
                low_bound = 1
            else:
                low_bound = 0
            print("processing: {}".format(dir_path))
            if len(files) < model_config.num_sequence+low_bound:
                filename = "{}/{}.jpg".format(dir_path, 0)
                for n in range(0, model_config.num_sequence + low_bound - len(files)):
                    c_img = cv2.imread(filename)
                    c_img = cv2.resize(c_img, (model_config.image_height, model_config.image_width))
                    img.append(c_img)
            print("total files: {}, number of padded images: {}".format(len(files), len(img)))
            for file_id in range(low_bound, min(len(files), model_config.num_sequence+low_bound)):
                filename = "{}/{}.jpg".format(dir_path, file_id)

                c_img = cv2.imread(filename)
                c_img = cv2.resize(c_img, (model_config.image_height, model_config.image_width))
                # plt.imshow(c_img, aspect="auto")
                # plt.show()
                img.append(c_img)
            print("total files: {}, number of read images: {}".format(len(files), len(img)))
            img = np.asarray(img)
            softmax_output = sess.run(
                fetches=["softmax:0"],
                feed_dict={
                    "image_feed:0": img,
                })
            prediction = np.argmax(softmax_output)
            res_file.write("{}, {}\n".format(dirname, prediction))


            # for file_id in range(1, model_config.num_sequence+1):
            #     dir_path = os.path.join(FLAGS.input_file_directory, dirname)
            #     filename = "{}/{}.jpg".format(dir_path, file_id)
            #     c_img = cv2.imread(filename)
            #     c_img = cv2.resize(c_img, (224, 224))
            #     #plt.imshow(c_img, aspect="auto")
            #     #plt.show()
            #     img.append(c_img)
            # img = np.asarray(img)
            # softmax_output = sess.run(
            #     fetches=["softmax:0"],
            #     feed_dict={
            #         "image_feed:0": img,
            #     })
            # prediction = np.argmax(softmax_output)
            # res_file.write("{}, {}\n".format(dirname, prediction))
    res_file.close()


if __name__ == "__main__":
  tf.app.run()