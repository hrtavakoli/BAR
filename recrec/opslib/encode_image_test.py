
# test file for the encode_image

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from encode_image import alexnet_encode

class AlexNetEncodeTest(tf.test.TestCase):

    def setUp(self):
        super(AlexNetEncodeTest, self).setUp()

        batch_size = 40
        height = 224
        width = 224
        num_channels = 3

        self._images = tf.placeholder(tf.float32,
                                      [batch_size, height, width, num_channels])
        self._batch_size = batch_size
        self._num_classes = 128



    def _verifyParameterCounts(self, endpoints):
        """Verifies the number of parameters in the inception model."""

        expected_params = [
            "AlexNet/conv1",
            "AlexNet/conv2",
            "AlexNet/pool2",
            "AlexNet/conv3",
            "AlexNet/pool3",
            "AlexNet/fc6",
            "AlexNet/fc7",
        ]
        self.assertSetEqual(set(expected_params), set(endpoints.keys()))

    def _assertCollectionSize(self, expected_size, collection):
        """ A helper function to ensure correct number of variables"""
        actual_size = len(tf.get_collection(collection))
        if expected_size != actual_size:
            self.fail("Found %d items in collection %s (expected %d)." %
                      (actual_size, collection, expected_size))

    def testNetworkEndPoints(self):
        with self.test_session():
            embeddings, endpoints = alexnet_encode(self._images, trainable=True, is_training=True)
            self._verifyParameterCounts(endpoints)

    def testNetworkVariables(self):
        expected_names = ['AlexNet/conv1/weights',
                          'AlexNet/conv1/biases',
                          'AlexNet/conv2/weights',
                          'AlexNet/conv2/biases',
                          'AlexNet/conv3/weights',
                          'AlexNet/conv3/biases',
                          'AlexNet/fc6/weights',
                          'AlexNet/fc6/biases',
                          'AlexNet/fc7/weights',
                          'AlexNet/fc7/biases',
                          ]
        with self.test_session():
            _, _ = alexnet_encode(self._images, trainable=True, is_training=True)
            model_variables = [v.op.name for v in tf.global_variables()]
            self.assertSetEqual(set(model_variables), set(expected_names))


    def testTrainableTrueIsTrainingTrue(self):
        with self.test_session():
            embeddings, endpoints = alexnet_encode(self._images, trainable=True, is_training=True)
            self.assertEqual([self._batch_size, self._num_classes], embeddings.get_shape().as_list())

            self._assertCollectionSize(10, tf.GraphKeys.GLOBAL_VARIABLES)
            self._assertCollectionSize(10, tf.GraphKeys.TRAINABLE_VARIABLES)
            self._assertCollectionSize(0, tf.GraphKeys.UPDATE_OPS)
            self._assertCollectionSize(0, tf.GraphKeys.REGULARIZATION_LOSSES)
            self._assertCollectionSize(0, tf.GraphKeys.LOSSES)
            self._assertCollectionSize(12, tf.GraphKeys.SUMMARIES)


    def testTrainableFalseIsTrainingTrue(self):
        with self.test_session():
            embeddings, endpoints = alexnet_encode(self._images, trainable=False, is_training=True)

            self.assertEqual([self._batch_size, self._num_classes], embeddings.get_shape().as_list())

            self._assertCollectionSize(10, tf.GraphKeys.GLOBAL_VARIABLES)
            self._assertCollectionSize(0, tf.GraphKeys.TRAINABLE_VARIABLES)
            self._assertCollectionSize(0, tf.GraphKeys.UPDATE_OPS)
            self._assertCollectionSize(0, tf.GraphKeys.REGULARIZATION_LOSSES)
            self._assertCollectionSize(0, tf.GraphKeys.LOSSES)
            self._assertCollectionSize(12, tf.GraphKeys.SUMMARIES)

    def testTrainableFalseIsTrainingFalse(self):
        with self.test_session():
            embeddings, endpoints = alexnet_encode(self._images, trainable=False, is_training=False)

            self.assertEqual([self._batch_size, self._num_classes], embeddings.get_shape().as_list())
            self._assertCollectionSize(10, tf.GraphKeys.GLOBAL_VARIABLES)
            self._assertCollectionSize(0, tf.GraphKeys.TRAINABLE_VARIABLES)
            self._assertCollectionSize(0, tf.GraphKeys.UPDATE_OPS)
            self._assertCollectionSize(0, tf.GraphKeys.REGULARIZATION_LOSSES)
            self._assertCollectionSize(0, tf.GraphKeys.LOSSES)
            self._assertCollectionSize(12, tf.GraphKeys.SUMMARIES)



if __name__ == "__main__":
    tf.test.main()