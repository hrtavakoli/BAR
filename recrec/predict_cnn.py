from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import os
import cv2


from recrec import imgRecModel
from recrec import configuration

FLAGS = tf.app.flags.FLAGS

#tf.flags.DEFINE_string("input_file_directory", "./data/test_img/",
#tf.flags.DEFINE_string("input_file_directory", "./data/test_data/img_human_1/",
tf.flags.DEFINE_string("input_file_directory", "/home/rtavah1/projects/databases/imgNet/val/",
                       "File pattern of input files.")
#tf.flags.DEFINE_string("chk_model", "./data/train/imgrec/model.ckpt-493156",
tf.flags.DEFINE_string("chk_model", "./log_full_ff/model.ckpt-492356",
#tf.flags.DEFINE_string("chk_model", "./data/recognition_vgg/model.ckpt-10000",
                       "the checkpoint to load model")

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


    # Load the files in the input file directory
    file_list = os.listdir(FLAGS.input_file_directory)


    # Build the inference graph.
    g = tf.Graph()
    with g.as_default():
        # build the model for inference
        model = imgRecModel.ImgRecModel(model_config,
                                        mode="inference")
        model.build_inputs()
        model.build_model()
        restore_function = build_graph_from_checkpoint(FLAGS.chk_model)
    g.finalize()

    res_file = open("./results/result_HDNet_imgNet_ff_492k.txt", "w")

    # Start a session and start guessing image type
    with tf.Session(graph=g) as sess:
        restore_function(sess)

        for filename in file_list:
            img = cv2.imread(os.path.join(FLAGS.input_file_directory, filename, '0.jpg'))
            img = cv2.resize(img, (224, 224))
            softmax_output = sess.run(
                fetches=["softmax:0"],
                feed_dict={
                    "image_feed:0": img,
                })
            prediction = np.argmax(softmax_output)
            res_file.write("{}, {}\n".format(filename, prediction))

    res_file.close()


if __name__ == "__main__":
  tf.app.run()
