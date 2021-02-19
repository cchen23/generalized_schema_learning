"""Modules to create the LSTM-LN architecture.

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

from core_rnn_cell_impl import _linear

from tensorflow.python.ops.rnn_cell_impl import RNNCell

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
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell.py (branch r0.10 https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/python/ops/rnn_cell.py)
class BasicLSTMCell(RNNCell):
  """Basic LSTM recurrent network cell.
  The implementation is based on: http://arxiv.org/abs/1409.2329.
  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.
  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.
  For advanced models, please use the full LSTMCell that follows.
  """

  def __init__(self, num_units, forget_bias=1.0, input_size=None,
               state_is_tuple=False, activation=tanh):
    """Initialize the basic LSTM cell.
    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      input_size: Deprecated and unused.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  By default (False), they are concatenated
        along the column axis.  This default behavior will soon be deprecated.
      activation: Activation function of the inner states.
    """
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation

  @property
  def state_size(self):
    return (LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with vs.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
      # Parameters of gates are concatenated into one multiply for efficiency.
      if self._state_is_tuple:
        c, h = state
      else:
        c, h = array_ops.split(state, 2, axis=1)
      concat = _linear([inputs, h], 4 * self._num_units, True)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = array_ops.split(concat, 4, axis=1)

      i = ln(i, scope = 'i/')
      j = ln(j, scope = 'j/')
      f = ln(f, scope = 'f/')
      o = ln(o, scope = 'o/')
      new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) *
               self._activation(j))
      new_h = self._activation(new_c) * sigmoid(o)

      if self._state_is_tuple:
        new_state = LSTMStateTuple(new_c, new_h)
      else:
        new_state = array_ops.concat([new_c, new_h], 1)
      return new_h, new_state, concat


class lstmln_model(object):

    def __init__(self, FLAGS):
        self.embedding = tf.placeholder(tf.float32)
        self.X = tf.placeholder(tf.float32,
            shape=[None, FLAGS.input_dim, FLAGS.num_classes], name='inputs_X')
        self.y = tf.placeholder(tf.float32,
            shape=[None, FLAGS.num_classes], name='targets_y')
        self.l = tf.placeholder(tf.float32, [], # need [] for tf.scalar_mul
            name="learning_rate")
        self.e = tf.placeholder(tf.float32, [],
            name="decay_rate")
        self.gate_history = None
        self.hidden_history = None
        with tf.variable_scope("LSTMLN"):

            # softmax weights (proper initialization)
            self.W_softmax = tf.Variable(tf.random_uniform(
                [FLAGS.num_hidden_units, FLAGS.num_classes],
                -np.sqrt(2.0 / FLAGS.num_hidden_units),
                np.sqrt(2.0 / FLAGS.num_hidden_units)),
                dtype=tf.float32)
            self.b_softmax = tf.Variable(tf.zeros(
                [FLAGS.num_classes]),
                dtype=tf.float32)

        # LSTM.
        print('num hidden units', FLAGS.num_hidden_units)
        self.lstm_1 = BasicLSTMCell(FLAGS.num_hidden_units)
        self.lstm_2 = BasicLSTMCell(FLAGS.num_hidden_units)
        self.lstm_3 = BasicLSTMCell(FLAGS.num_hidden_units)
        self.new_state = self.lstm_1.zero_state(FLAGS.batch_size, tf.float32)
        with tf.variable_scope("lstmln_step") as scope:
            for t in range(0, FLAGS.input_dim):
                if t > 0:
                    scope.reuse_variables()
                self.new_h, self.new_state, concat = self.lstm_1(self.X[:, t, :], self.new_state, scope="lstm1")
                self.new_h, self.new_state, concat = self.lstm_2(self.X[:, t, :], self.new_state, scope="lstm2")
                self.new_h, self.new_state, concat = self.lstm_3(self.X[:, t, :], self.new_state, scope="lstm3")
                if t > 0:
                    self.gate_history = tf.concat([self.gate_history, tf.expand_dims(concat, axis=2)], axis=2)
                    self.hidden_history = tf.concat([self.hidden_history, tf.expand_dims(self.new_state, axis=2)], axis=2)
                else:
                    self.gate_history = tf.expand_dims(concat, axis=2)
                    self.hidden_history = tf.expand_dims(self.new_state, axis=2)
        # All inputs processed! Time for softmax
        self.logits = tf.matmul(self.new_h, self.W_softmax) + self.b_softmax

        # Loss
        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.y, predictions=self.logits)) # If embedding instead of one-hot, don't want softmax (want actual values at each index, not just a classifier).

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
        self.accuracy = embedding_util.get_01_accuracy(self.y, self.logits, self.embedding)

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
            run_option: Trains if `backprop`, tests and gives loss and accuracy if `forward_only`, tests and gives loss, accuracy, and state histories if `analyze`.

        Returns:
            If 'analyze': The loss, accuracy, and network states.
            If backprop: The loss, accuracy, gradient norm, and an Optimizer that applies the gradient.
            If forward_only: The loss and accuracy.
        """
        input_feed = {self.X: batch_X, self.y: batch_y, self.embedding: batch_embedding, self.l:l, self.e:e}

        if run_option == "backprop": # training
            output_feed = [self.loss, self.accuracy, self.norm,
            self.update]
        elif run_option == "forward_only": # testing
            output_feed = [self.loss, self.accuracy]
        elif run_option == "analyze":
            output_feed = [self.loss, self.accuracy, (self.gate_history, self.hidden_history)]
        else:
            raise ValueError("Invalid run_option.")
        # process outputs
        outputs = sess.run(output_feed, input_feed)

        if run_option == "analyze":
            return outputs[0], outputs[1], outputs[2], None
        elif run_option == "backprop":
            return outputs[0], outputs[1], outputs[2], outputs[3]
        elif run_option == "forward_only":
            return outputs[0], outputs[1]
