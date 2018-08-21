

"""Recognition configuration files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class CNNModelConfig(object):
  """Wrapper class for model hyperparameters."""

  def __init__(self):
    """Sets the default model hyperparameters."""
    # File pattern of sharded TFRecord file containing SequenceExample protos.
    # Must be provided in training and evaluation modes.
    self.input_file_pattern = None

    # which network to encode
    self.CNN_encoder = None

    # weather to fine tune the CNN
    self.fine_tune_cnn = None

    # where to load the CNN checkpoint
    self.CNN_ckpt = None

    # Approximate number of values per input shard. Used to ensure sufficient
    # mixing between shards in training.
    #self.values_per_input_shard = 4703
    self.values_per_input_shard = 1200
    # Minimum number of shards to keep in the input queue.
    self.input_queue_capacity_factor = 2
    # Number of threads for prefetching protos.
    self.num_input_reader_threads = 2

    # Number of threads for image preprocessing. Should be a multiple of 2.
    self.num_preprocess_threads = 2

    # Batch size.
    self.batch_size = 5
    #self.batch_size = 50

    # Dimensions of  input images.
    # dimensions for HDNet
    self.image_height = 224
    self.image_width = 224
    # dimensions for ResNet
    #self.image_height = 240
    #self.image_width = 240

    # 3rd image dimension, 3 for color images
    self.color = 3

    # Scale used to initialize model variables.
    #self.initializer_scale = 0.25
    self.initializer_scale = 0.05

    # How many classes of object to classify
    self.num_classes = 101
    #self.num_classes = 100

    # number of the images in the sequence
    self.num_sequence = 3

    # If < 1.0, the dropout keep probability applied to drop out.
    self.dropout_rate = 0.85


class CNNTrainingConfig(object):
  """Wrapper class for training hyperparameters."""

  def __init__(self):
    """Sets the default training hyperparameters."""
    # Number of examples per epoch of training data.
    self.num_examples_per_epoch = 129400#2800#4073


    # Optimizer for training the model.
    self.optimizer = "Adam"

    # Initial learning rate.
    self.initial_learning_rate = 0.00001
    #self.initial_learning_rate = 0.0001
    #self.initial_learning_rate = 0.01

    # How many model checkpoints to keep.
    self.max_checkpoints_to_keep = 5