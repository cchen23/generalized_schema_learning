"""Modules to create the Fastweights, CONTROL, and RNN-LN architectures.

NOTE: Adapted from https://github.com/GokuMohandas/fast-weights.
"""
import numpy as np
import sys
import tensorflow as tf

sys.path.append("../")
import embedding_util

from custom_GRU import (
    ln,
)

class fast_weights_model(object):

    def __init__(self, FLAGS):
        epsilon = 1e-5
        self.embedding = tf.placeholder(tf.float32)
        self.X = tf.placeholder(tf.float32,
            shape=[None, FLAGS.input_dim, FLAGS.num_classes], name='inputs_X')
        self.y = tf.placeholder(tf.float32,
            shape=[None, FLAGS.num_classes], name='targets_y')
        self.l = tf.placeholder(tf.float32, [], # need [] for tf.scalar_mul
            name="learning_rate")
        self.e = tf.placeholder(tf.float32, [],
            name="decay_rate")
        self.hidden_history = None
        self.fastweights_history = None
        self.model_name = FLAGS.model_name

        with tf.variable_scope("fast_weights"):

            # input weights (proper initialization)
            self.W_x = tf.Variable(tf.random_uniform(
                [FLAGS.num_classes, FLAGS.num_hidden_units],
                -np.sqrt(2.0/FLAGS.num_classes),
                np.sqrt(2.0/FLAGS.num_classes)),
                dtype=tf.float32)
            self.b_x = tf.Variable(tf.zeros(
                [FLAGS.num_hidden_units]),
                dtype=tf.float32)

            # hidden weights (See Hinton's video @ 21:20)
            self.W_h = tf.Variable(
                initial_value=0.05 * np.identity(FLAGS.num_hidden_units),
                dtype=tf.float32)

            # softmax weights (proper initialization)
            self.W_softmax = tf.Variable(tf.random_uniform(
                [FLAGS.num_hidden_units, FLAGS.num_classes],
                -np.sqrt(2.0 / FLAGS.num_hidden_units),
                np.sqrt(2.0 / FLAGS.num_hidden_units)),
                dtype=tf.float32)
            self.b_softmax = tf.Variable(tf.zeros(
                [FLAGS.num_classes]),
                dtype=tf.float32)

            # scale and shift for layernorm
            self.gain = tf.Variable(tf.ones(
                [FLAGS.num_hidden_units]),
                dtype=tf.float32)
            self.bias = tf.Variable(tf.zeros(
                [FLAGS.num_hidden_units]),
                dtype=tf.float32)

        # fast weights and hidden state initialization
        self.A = tf.zeros(
            [FLAGS.batch_size, FLAGS.num_hidden_units, FLAGS.num_hidden_units],
            dtype=tf.float32)
        self.h = tf.zeros(
            [FLAGS.batch_size, FLAGS.num_hidden_units],
            dtype=tf.float32)

        # NOTE:inputs are batch-major
        # Process batch by time-major
        for t in range(0, FLAGS.input_dim):

            # hidden state (preliminary vector)
            self.h = tf.nn.tanh((tf.matmul(self.X[:, t, :], self.W_x)+self.b_x) +
                (tf.matmul(self.h, self.W_h)))

            # Forward weight and layer normalization
            if self.model_name in ['RNN-LN-FW']:

                # Reshape h to use with a
                self.h_s = tf.reshape(self.h,
                    [FLAGS.batch_size, 1, FLAGS.num_hidden_units])

                # Create the fixed A for this time step
                self.A = tf.add(tf.scalar_mul(self.l, self.A),
                    tf.scalar_mul(self.e, tf.matmul(tf.transpose(
                        self.h_s, [0, 2, 1]), self.h_s)))

                # Loop for S steps
                for _ in range(FLAGS.S):
                    self.h_s = tf.squeeze(tf.reshape(
                        tf.matmul(self.X[:, t, :], self.W_x)+self.b_x,
                        tf.shape(self.h_s)) + tf.reshape(
                        tf.matmul(self.h, self.W_h), tf.shape(self.h_s)) + \
                        tf.matmul(self.h_s, self.A))

                    # Apply layernorm
                    mu = tf.expand_dims(tf.reduce_mean(self.h_s, reduction_indices=1), 1) # each sample
                    sigma = tf.expand_dims(tf.sqrt(tf.reduce_mean(tf.square(self.h_s - mu),
                        reduction_indices=1)), 1)
                    self.h_s = tf.div(tf.multiply(self.gain, (self.h_s - mu)), sigma + epsilon) + \
                        self.bias
                    # Apply nonlinearity
                    self.h_s = tf.nn.tanh(self.h_s)

                # Reshape h_s into h
                self.h = tf.reshape(self.h_s,
                    [FLAGS.batch_size, FLAGS.num_hidden_units])

                # Save history info
                if t > 0:
                    self.hidden_history = tf.concat([self.hidden_history, tf.expand_dims(self.h, axis=2)], axis=2)
                    self.fastweights_history = tf.concat([self.fastweights_history, tf.expand_dims(self.A, axis=3)], axis=3)
                else:
                    self.hidden_history = tf.expand_dims(self.h, axis=2)
                    self.fastweights_history = tf.expand_dims(self.A, axis=3)

            elif self.model_name == 'RNN-LN': # no fast weights but still LN
                # Apply layer norm
                with tf.variable_scope('just_ln') as scope:
                    if t > 0:
                        scope.reuse_variables()
                    self.h = ln(self.h, scope='h/')
                # Save history info
                if t > 0:
                    self.hidden_history = tf.concat([self.hidden_history, tf.expand_dims(self.h, axis=2)], axis=2)
                else:
                    self.hidden_history = tf.expand_dims(self.h, axis=2)

            elif self.model_name == 'CONTROL': # no fast weights or LN
                if t > 0:
                    self.hidden_history = tf.concat([self.hidden_history, tf.expand_dims(self.h, axis=2)], axis=2)
                else:
                    self.hidden_history = tf.expand_dims(self.h, axis=2)

        # All inputs processed! Time for softmax
        self.logits = tf.matmul(self.h, self.W_softmax) + self.b_softmax

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
            output_feed = [self.loss, self.accuracy, self.norm, self.update]
        elif run_option == "forward_only": # testing
            output_feed = [self.loss, self.accuracy]
        elif run_option == "analyze":
            output_feed = [self.loss, self.accuracy, self.hidden_history, self.fastweights_history]
        else:
            raise ValueError("Invalid run option.")

        # process outputs
        outputs = sess.run(output_feed, input_feed)

        if run_option == "backprop":
            return outputs[0], outputs[1], outputs[2], outputs[3]
        elif run_option == "forward_only":
            return outputs[0], outputs[1]
        elif run_option == "analyze":
            return outputs[0], outputs[1], (None, outputs[2]), outputs[3]
