# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NOTE: Adapted from https://github.com/deepmind/dnc.
# ==============================================================================
"""Reduced NTM access modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import sonnet as snt
import tensorflow as tf

import addressing
import util

AccessState = collections.namedtuple('AccessState', (
    'memory', 'read_weights', 'write_weights'))


def _erase_and_write(memory, address, reset_weights, values):
  """Module to erase and write in the external memory.

  Erase operation:
    M_t'(i) = M_{t-1}(i) * (1 - w_t(i) * e_t)

  Add operation:
    M_t(i) = M_t'(i) + w_t(i) * a_t

  where e are the reset_weights, w the write weights and a the values.

  Args:
    memory: 3-D tensor of shape `[batch_size, memory_size, word_size]`.
    address: 3-D tensor `[batch_size, num_writes, memory_size]`.
    reset_weights: 3-D tensor `[batch_size, num_writes, word_size]`.
    values: 3-D tensor `[batch_size, num_writes, word_size]`.

  Returns:
    3-D tensor of shape `[batch_size, num_writes, word_size]`.
  """
  with tf.name_scope('erase_memory', values=[memory, address, reset_weights]):
    expand_address = tf.expand_dims(address, 3)
    reset_weights = tf.expand_dims(reset_weights, 2)
    weighted_resets = expand_address * reset_weights
    reset_gate = tf.reduce_prod(1 - weighted_resets, [1])
    memory *= reset_gate

  with tf.name_scope('additive_write', values=[memory, address, values]):
    add_matrix = tf.matmul(address, values, adjoint_a=True)
    memory += add_matrix

  return memory


class MemoryAccess(snt.RNNCore):
  """Access module of the Neural Turing Machine.

  This memory module supports multiple read and write heads.

  Write-address selection is done by an interpolation between content-based
  lookup and using unused memory.

  Read-address selection is done by an interpolation of content-based lookup
  and following the link graph in the forward or backwards read direction.
  """

  def __init__(self,
               memory_size=128,
               word_size=20,
               num_reads=1,
               num_writes=1,
               name='memory_access'):
    """Creates a MemoryAccess module.

    Args:
      memory_size: The number of memory slots (N in the DNC paper).
      word_size: The width of each memory slot (W in the DNC paper)
      num_reads: The number of read heads (R in the DNC paper).
      num_writes: The number of write heads (fixed at 1 in the paper).
      name: The name of the module.
    """
    super(MemoryAccess, self).__init__(name=name)
    self._memory_size = memory_size
    self._word_size = word_size
    self._num_reads = num_reads
    self._num_writes = num_writes
    self._num_heads = num_reads + num_writes
    self._write_content_weights_mod = addressing.CosineWeights(
        num_writes, word_size, name='write_content_weights')
    self._read_content_weights_mod = addressing.CosineWeights(
        num_reads, word_size, name='read_content_weights')
    SHIFT_RANGE = 1

  def _build(self, inputs, prev_state):
    """Connects the MemoryAccess module into the graph.

    Args:
      inputs: tensor of shape `[batch_size, input_size]`. This is used to
          control this access module.
      prev_state: Instance of `AccessState` containing the previous state.

    Returns:
      A tuple `(output, next_state)`, where `output` is a tensor of shape
      `[batch_size, num_reads, word_size]`, and `next_state` is the new
      `AccessState` named tuple at the current time t.
    """
    inputs = self._read_inputs(inputs)

    # Write to memory.
    write_weights = self._write_weights(inputs, prev_state.memory, prev_write_weights=prev_state.write_weights)
    memory = _erase_and_write(
        prev_state.memory,
        address=write_weights,
        reset_weights=inputs['erase_vectors'],
        values=inputs['write_vectors'])

    # Read from memory.
    read_weights = self._read_weights(
        inputs,
        memory=memory,
        prev_read_weights=prev_state.read_weights)
    read_words = tf.matmul(read_weights, memory)

    return (read_words, AccessState(
        memory=memory,
        read_weights=read_weights,
        write_weights=write_weights))

  def _read_inputs(self, inputs):
    """Applies transformations to `inputs` to get control for this module."""

    def _linear(first_dim, second_dim, name, activation=None):
      """Returns a linear transformation of `inputs`, followed by a reshape."""
      linear = snt.Linear(first_dim * second_dim, name=name)(inputs)
      if activation is not None:
        linear = activation(linear, name=name + '_activation')
      return tf.reshape(linear, [-1, first_dim, second_dim])

    # v_t^i - The vectors to write to memory, for each write head `i`.
    write_vectors = _linear(self._num_writes, self._word_size, 'write_vectors')

    # e_t^i - Amount to erase the memory by before writing, for each write head.
    erase_vectors = _linear(self._num_writes, self._word_size, 'erase_vectors',
                            tf.sigmoid)

    # g_t - Blend between the weighting produced by the head at the previous
    # timestep and the weighting w_t^c produced at the current time-step.
    interpolation_gate_write = tf.sigmoid(
        snt.Linear(self._num_writes, name='interpolation_gate_write')(inputs))
    interpolation_gate_read = tf.sigmoid(
        snt.Linear(self._num_reads, name='interpolation_gate_read')(inputs))

    # s_t - Shift weighting that defines a normalized distribution over the
    # allowable integer shifts.
    # NOTE: Reduced NTM has no shift weighitng.
    # shift_weighting_write = tf.sigmoid(
    #     snt.Linear(self._num_writes * (SHIFT_RANGE * 2 + 1), name='shift_weighting_write')(inputs))
    # shift_weighting_read = tf.sigmoid(
    #     snt.Linear(self._num_reads * (SHIFT_RANGE * 2 + 1), name='shift_weighting_read')(inputs))

    # gamma_t - Scalar to sharpen the final weighting
    gamma_write = tf.add(tf.nn.softplus(
        snt.Linear(self._num_writes, name='gamma_write')(inputs)), tf.constant(1.0))
    gamma_read = tf.add(tf.nn.softplus(
        snt.Linear(self._num_reads, name='gamma_read')(inputs)), tf.constant(1.0))

    # Parameters for the (read / write) "weights by content matching" modules.
    write_keys = _linear(self._num_writes, self._word_size, 'write_keys')
    write_strengths = snt.Linear(self._num_writes, name='write_strengths')(
        inputs)

    read_keys = _linear(self._num_reads, self._word_size, 'read_keys')
    read_strengths = snt.Linear(self._num_reads, name='read_strengths')(inputs)

    result = {
        'read_content_keys': read_keys,
        'read_content_strengths': read_strengths,
        'write_content_keys': write_keys,
        'write_content_strengths': write_strengths,
        'write_vectors': write_vectors,
        'erase_vectors': erase_vectors,
        'interpolation_gate_write': interpolation_gate,
        # 'shift_weighting_write': shift_weighting, NOTE: Reduced NTM has no shift weighting.
        'gamma_write': gamma,
        'interpolation_gate_read': interpolation_gate,
        # 'shift_weighting_read': shift_weighting, NOTE: Reduced NTM has no shift weighting.
        'gamma_read': gamma,
    }
    return result

  def _write_weights(self, inputs, memory, prev_write_weights):
    """Calculates the memory locations to write to.

    This uses a combination of content-based lookup and finding an unused
    location in memory, for each write head.

    Args:
      inputs: Collection of inputs to the access module, including controls for
          how to chose memory writing, such as the content to look-up and the
          weighting between content-based and allocation-based addressing.
      memory: A tensor of shape  `[batch_size, memory_size, word_size]`
          containing the current memory contents.
      prev_write_weights: A tensor of shape `[batch_size, num_writes,
          memory_size]` containing the previous read locations.

    Returns:
      tensor of shape `[batch_size, num_writes, memory_size]` indicating where
          to write to (if anywhere) for each write head.
    """
    with tf.name_scope('write_weights', values=[inputs, memory, usage]):
      # c_t^{w, i} - The content-based weights for each write head.
      write_content_weights = self._write_content_weights_mod(
          memory, inputs['write_content_keys'],
          inputs['write_content_strengths'])

      # w_t^g
      interpolation_gate = tf.expand_dims(inputs['interpolation_gate_write'],1)
      gated_weighting = interpolation_gate * content_weights + (1.0 - interpolation_gate) * prev_write_weights

      # w_t_bar
      # shift_weighting = tf.reshape(inputs['shift_weighting_write'], #  NOTE: Reduced NTM has no shift weighting.
      w_bar = gated_weighting
      gamma = tf.expand_dims(inputs['gamma_write'],1)
      w_bar_gamma = tf.pow(w_bar, gamma)
      w = w_bar_gamma / tf.reduce_sum(w_bar_gamma)
      # w_t^{w, i} - The write weightings for each write head.
      return w_bar_gamma

  def _read_weights(self, inputs, memory, prev_read_weights):
    """Calculates read weights for each read head.

    The read weights are a combination of following the link graphs in the
    forward or backward directions from the previous read position, and doing
    content-based lookup. The interpolation between these different modes is
    done by `inputs['read_mode']`.

    Args:
      inputs: Controls for this access module. This contains the content-based
          keys to lookup, and the weightings for the different read modes.
      memory: A tensor of shape `[batch_size, memory_size, word_size]`
          containing the current memory contents to do content-based lookup.
      prev_read_weights: A tensor of shape `[batch_size, num_reads,
          memory_size]` containing the previous read locations.

    Returns:
      A tensor of shape `[batch_size, num_reads, memory_size]` containing the
      read weights for each read head.
    """
    with tf.name_scope(
        'read_weights', values=[inputs, memory, prev_read_weights]):
      # c_t^{r, i} - The content weightings for each read head.
      content_weights = self._read_content_weights_mod(
          memory, inputs['read_content_keys'], inputs['read_content_strengths'])

      # w_t^g
      interpolation_gate = tf.expand_dims(inputs['interpolation_gate_read'],1)
      gated_weighting = interpolation_gate * content_weights + (1.0 - interpolation_gate) * prev_write_weights

      # w_t_bar
      # shift_weighting = tf.reshape(inputs['shift_weighting_read'], # NOTE: Reduced NTM has no shift weighting.
      w_bar = gated_weighting
      gamma = tf.expand_dims(inputs['gamma_read'],1)
      w_bar_gamma = tf.pow(w_bar, gamma)
      w = w_bar_gamma / tf.reduce_sum(w_bar_gamma)
      return w

  @property
  def state_size(self):
    """Returns a tuple of the shape of the state tensors."""
    return AccessState(
        memory=tf.TensorShape([self._memory_size, self._word_size]),
        read_weights=tf.TensorShape([self._num_reads, self._memory_size]),
        write_weights=tf.TensorShape([self._num_writes, self._memory_size]))

  @property
  def output_size(self):
    """Returns the output shape."""
    return tf.TensorShape([self._num_reads, self._word_size])
