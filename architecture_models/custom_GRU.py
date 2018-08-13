"""Modules to create the GRU-LN architecture.

NOTE: Adapted from https://github.com/GokuMohandas/fast-weights.
"""
import collections
import math
import numpy as np
import sys
import tensorflow as tf

sys.path.append("../")
import embedding_util

from tensorflow.python.framework import (
    ops,
    tensor_shape,
)

from tensorflow.python.ops import (
    array_ops,
    clip_ops,
    embedding_ops,
    init_ops,
    math_ops,
    nn_ops,
    partitioned_variables,
    variable_scope as vs,
)

from tensorflow.python.ops.math_ops import (
    sigmoid,
    tanh,
)

from tensorflow.python.platform import (
    tf_logging as logging,
)

from tensorflow.python.util import (
    nest,
)

from core_rnn_cell_impl import _linear

# LN funcition
def ln(inputs, epsilon=1e-5, scope=None):

    """ Computer LN given an input tensor. We get in an input of shape
    [N X D] and with LN we compute the mean and var for each individual
    training point across all it's hidden dimensions rather than across
    the training batch as we do in BN. This gives us a mean and var of shape
    [N X 1].
    """
    mean, var = tf.nn.moments(inputs, [1], keep_dims=True)
    with tf.variable_scope(scope + 'LN'):
        scale = tf.get_variable('alpha',
                                shape=[inputs.get_shape()[1]],
                                initializer=tf.constant_initializer(1))
        shift = tf.get_variable('beta',
                                shape=[inputs.get_shape()[1]],
                                initializer=tf.constant_initializer(0))
    LN = scale * (inputs - mean) / tf.sqrt(var + epsilon) + shift

    return LN

# Modified from:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell.py
class GRUCell(tf.contrib.rnn.RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self, num_units, input_size=None, activation=tanh, keep_prob=None):
        if input_size is not None:
          logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._activation = activation
        self._keep_prob = keep_prob

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        with vs.variable_scope(scope or type(self).__name__):  # "GRUCell"
          with vs.variable_scope("Gates"):  # Reset gate and update gate.
            # We start with bias of 1.0 to not reset and not update.
            r, u = array_ops.split(_linear([inputs, state],
                           2 * self._num_units, True, 1.0), 2, axis=1)

            # Apply Layer Normalization to the two gates
            r = ln(r, scope = 'r/')
            u = ln(r, scope = 'u/')

            r, u = sigmoid(r), sigmoid(u)
          with vs.variable_scope("Candidate"):
            if self._keep_prob == 1:
              c = self._activation(_linear([inputs, r * state],
                                         self._num_units, True))
            else:
              c = tf.nn.dropout(self._activation(_linear([inputs, r * state],
                                         self._num_units, True)), keep_prob=self._keep_prob)
          new_h = u * state + (1 - u) * c
        return new_h, new_h, tf.concat([r,u], axis=0)

class gru_model(object):

    def __init__(self, FLAGS):
        self.vector_type = FLAGS.vector_type
        self.embedding = tf.placeholder(tf.float32)
        self.X = tf.placeholder(tf.float32,
            shape=[None, FLAGS.input_dim, FLAGS.num_classes], name='inputs_X')
        self.y = tf.placeholder(tf.float32,
            shape=[None, FLAGS.num_classes], name='targets_y')
        self.l = tf.placeholder(tf.float32, [], # need [] for tf.scalar_mul
            name="learning_rate")
        self.e = tf.placeholder(tf.float32, [],
            name="decay_rate")
        nobias = "nobias" in FLAGS.model_name
        keep_prob = FLAGS.keep_prob
        with tf.variable_scope("GRU"):

            # softmax weights (proper initialization)
            self.W_softmax = tf.Variable(tf.random_uniform(
                [FLAGS.num_hidden_units, FLAGS.num_classes],
                -np.sqrt(2.0 / FLAGS.num_hidden_units),
                np.sqrt(2.0 / FLAGS.num_hidden_units)),
                dtype=tf.float32)
            self.b_softmax = tf.Variable(tf.zeros(
                [FLAGS.num_classes]),
                dtype=tf.float32)

        self.h = tf.zeros(
            [FLAGS.batch_size, FLAGS.num_hidden_units],
            dtype=tf.float32)

        # GRU
        self.gru = GRUCell(FLAGS.num_hidden_units, keep_prob=keep_prob)
        with tf.variable_scope("gru_step") as scope:
            for t in range(0, FLAGS.input_dim):
                if t > 0:
                    scope.reuse_variables()
                self.outputs, self.h, concat = self.gru(self.X[:, t, :], self.h)
                if t > 0:
                    self.gate_history = tf.concat([self.gate_history, tf.expand_dims(concat, axis=2)], axis=2)
                    self.hidden_history = tf.concat([self.hidden_history, tf.expand_dims(self.h, axis=2)], axis=2)
                else:
                    self.gate_history = tf.expand_dims(concat, axis=2)
                    self.hidden_history = tf.expand_dims(self.h, axis=2)
        # All inputs processed! Time for softmax
        if nobias:
            print("NO BIAS IN FINAL LAYER")
            self.logits = tf.matmul(self.h, self.W_softmax)
        else:
            self.logits = tf.matmul(self.h, self.W_softmax) + self.b_softmax

        # Loss
        if FLAGS.vector_type in ["embedding_random", "embedding_random_newfillereachex"]:
            self.loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.y, predictions=self.logits)) # If embedding instead of one-hot, don't want softmax (want actual values at each index, not just a classifier).
        elif FLAGS.vector_type in ["ONEHOT"]:
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))

        # Optimization
        self.lr = tf.Variable(0.0, trainable=False)
        self.trainable_vars = tf.trainable_variables()
        # clip the gradient to avoid vanishing or blowing up gradients
        self.grads, self.norm = tf.clip_by_global_norm(
            tf.gradients(self.loss, self.trainable_vars), FLAGS.max_gradient_norm)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.update = optimizer.apply_gradients(
            zip(self.grads, self.trainable_vars))

        # Accuracy
        if FLAGS.vector_type in ["embedding_random", "embedding_random_newfillereachex"]:
            self.accuracy = embedding_util.get_01_accuracy(self.y, self.logits, self.embedding)
        elif FLAGS.vector_type in ["ONEHOT"]:
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1),
                tf.argmax(self.y, 1)), tf.float32))

        # Components for model saving
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=4)

    def step(self, sess, batch_X, batch_y, batch_embedding, l, e, run_option):
        """Run the network on a given batch.

        Args:
            sess: The current tensorflow session.
            batch_X: [batch_size x num_words_per_input x num_dimensions_per_word] matrix of inputs.
            batch_y: [batch_size x num_dimensions_per_word] matrix of correct outputs.
            batch_embedding: [num_words_in_corpus x num_dimensions_per_word] matrix of vector word representations.
            l: The learning rate.
            e: The decay rate.
            forward_only: If True, only runs the forward pass. If False, also performs gradient descent.

        Returns:
            If not forward_only: The loss, accuracy, gradient norm, and an Optimizer that applies the gradient.
            If forward_only: The loss and accuracy.
        """
        input_feed = {self.X: batch_X, self.y: batch_y, self.embedding: batch_embedding, self.l:l, self.e:e}

        if run_option == "backprop": # training
            output_feed = [self.loss, self.accuracy, self.norm,
            self.update]
        elif run_option == "foward_only": # validation
            output_feed = [self.loss, self.accuracy]
        elif run_option == "analyze":
            output_feed = [self.loss, self.accuracy, (self.gate_history, self.hidden_history)]
        else:
            raise ValueError("Incorrect run_option. Must be `backprop`, `forward_only`, or `analyze`")
        # process outputs
        outputs = sess.run(output_feed, input_feed)

        if run_option == "analyze":
            return outputs[0], outputs[1], outputs[2], outputs[3], None
        elif run_option == "backprop":
            return outputs[0], outputs[1], outputs[2], outputs[3]
        elif run_option == "forward_only":
            return outputs[0], outputs[1]
