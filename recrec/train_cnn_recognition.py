
"""Train the recognition model from scratch"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf


from recrec import imgRecModel
from recrec import configuration

FLAGS = tf.app.flags.FLAGS

#tf.flags.DEFINE_string("input_file_pattern", "./data/tfrecs/train_human_1.tfrecords",
#tf.flags.DEFINE_string("input_file_pattern", "/home/rtavah1/projects/databases/imgNet/tfrecs/Imagenet12-?????-of-00150",
tf.flags.DEFINE_string("input_file_pattern", "/home/rtavah1/projects/databases/imgNet/tfrecs/ImageNet12-?????-of-00150",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("train_dir", "./log_full/",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_boolean("train_cnn", True,
                        "Whether to train the cnn from scratch.")
tf.flags.DEFINE_integer("number_of_steps", 492356, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 100,
                        "Frequency at which loss and global step are logged.")

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
  assert FLAGS.input_file_pattern, "--input_file_pattern is required"
  assert FLAGS.train_dir, "--train_dir is required"

  model_config = configuration.CNNModelConfig()
  model_config.input_file_pattern = FLAGS.input_file_pattern
  training_config = configuration.CNNTrainingConfig()

  #model_config.CNN_ckpt = "./data/vgg_model/vgg_16.ckpt"
  model_config.CNN_encoder = "HDNet"
  model_config.fine_tune_cnn = True
  model_config.num_classes = 101

  # Create training directory.
  train_dir = FLAGS.train_dir
  if not tf.gfile.IsDirectory(train_dir):
    tf.logging.info("Creating training directory: %s", train_dir)
    tf.gfile.MakeDirs(train_dir)

  # Build the TensorFlow graph.
  g = tf.Graph()
  with g.as_default():
    # Build the model.
    model = imgRecModel.ImgRecModel(model_config, mode="train")
    model.build()

    # Set up the training ops.
    train_op = tf.contrib.layers.optimize_loss(
        loss=model.total_loss,
        global_step=model.global_step,
        learning_rate=training_config.initial_learning_rate,
        optimizer=training_config.optimizer)

    # Set up the Saver for saving and restoring model checkpoints.
    saver = tf.train.Saver(max_to_keep=training_config.max_checkpoints_to_keep)

  # Run training.
  tf.contrib.slim.learning.train(
      train_op,
      train_dir,
      log_every_n_steps=FLAGS.log_every_n_steps,
      graph=g,
      global_step=model.global_step,
      number_of_steps=FLAGS.number_of_steps,
      saver=saver)


if __name__ == "__main__":
  tf.app.run()