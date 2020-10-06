"""Modules to create the reduced NTM architecture (defined in ntm2/).

NOTE: Adapted from https://github.com/GokuMohandas/fast-weights.
"""
import collections
import math
import numpy as np
import tensorflow as tf

import sys
sys.path.append("../")
from directories import base_dir
import embedding_util
sys.path.append(base_dir + "architecture_models/ntm2_core")
import ntm2

class ntm2_model(object):

    def __init__(self, FLAGS):
        FLAGS.hidden_size = 50
        FLAGS.memory_size = 128
        FLAGS.word_size = 50
        FLAGS.num_write_heads = 1
        FLAGS.num_read_heads = 4
        FLAGS.clip_value = 20
        self.embedding = tf.placeholder(tf.float32)
        self.X = tf.placeholder(tf.float32,
            shape=[None, FLAGS.input_dim, FLAGS.num_classes], name='inputs_X')
        self.y = tf.placeholder(tf.float32,
            shape=[None, FLAGS.num_classes], name='targets_y')
        self.l = tf.placeholder(tf.float32, [], # need [] for tf.scalar_mul
            name="learning_rate")
        self.e = tf.placeholder(tf.float32, [],
            name="decay_rate")

        self.memory_history = None
        self.hidden_history = None
        self.gate_history = None

        with tf.variable_scope("NTM2"):
            # softmax weights (proper initialization)
            self.W_softmax = tf.Variable(tf.random_uniform(
                [FLAGS.num_hidden_units, FLAGS.num_classes],
                -np.sqrt(2.0 / FLAGS.num_hidden_units),
                np.sqrt(2.0 / FLAGS.num_hidden_units)),
                dtype=tf.float32)
            self.b_softmax = tf.Variable(tf.zeros(
                [FLAGS.num_classes]),
                dtype=tf.float32)

        # NTM2.
        access_config = {
            "memory_size": FLAGS.memory_size,
            "word_size": FLAGS.word_size,
            "num_reads": FLAGS.num_read_heads,
            "num_writes": FLAGS.num_write_heads,
        }
        controller_config = {
            "hidden_size": FLAGS.hidden_size,
        }
        clip_value = FLAGS.clip_value

        self.ntm2_core = ntm2.NTM2(access_config, controller_config, FLAGS.num_hidden_units, clip_value)
        self.ntm2state = self.ntm2_core.initial_state(FLAGS.batch_size)

        with tf.variable_scope("ntm2_step") as scope:
            for t in range(0, FLAGS.input_dim):
                if t > 0:
                    scope.reuse_variables()
                self.new_output, self.ntm2state, self.gates = self.ntm2_core(self.X[:, t, :], self.ntm2state)

                # Save history info
                if t > 0:
                    self.gate_history = tf.concat([self.gate_history, tf.expand_dims(self.gates, axis=2)], axis=2)
                    self.hidden_history = tf.concat([self.hidden_history, tf.expand_dims(self.ntm2state.controller_state.hidden, axis=2)], axis=2)
                    self.memory_history = tf.concat([self.memory_history, tf.expand_dims(self.ntm2state.access_state.memory, axis=3)], axis=3)
                else:
                    self.gate_history = tf.expand_dims(self.gates, axis=2)
                    self.hidden_history = tf.expand_dims(self.ntm2state.controller_state.hidden, axis=2)
                    self.memory_history = tf.expand_dims(self.ntm2state.access_state.memory, axis=3)

        # All inputs processed! Time for softmax
        self.logits = tf.matmul(self.new_output, self.W_softmax) + self.b_softmax

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
            output_feed = [self.loss, self.accuracy, (self.gate_history, self.hidden_history), self.memory_history]
        else:
            raise ValueError("Invalid run_option.")
        # process outputs
        outputs = sess.run(output_feed, input_feed)

        if run_option == "analyze":
            return outputs[0], outputs[1], outputs[2], outputs[3]
        elif run_option == "backprop":
            return outputs[0], outputs[1], outputs[2], outputs[3]
        elif run_option == "forward_only":
            return outputs[0], outputs[1]
