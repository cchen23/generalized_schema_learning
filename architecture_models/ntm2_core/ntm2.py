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

# NOTE: ADAPTED FROM https://github.com/deepmind/dnc.
# ==============================================================================
"""Reduced NTM Cores.

These modules create a reduced NTM core. They take input, pass parameters to the
memory access module, and integrate the output of memory to form an output. This
model is reduced because it does not contain the shift weights that allow a standard
NTM to link consecutive memory slots.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import sonnet as snt
import tensorflow as tf

import sys
sys.path.append("/home/cc27/Thesis/contextual_learning_csw/architecture_models/")
from architecture_models.sonnet_gated_rnn import LSTM

import access

NTM2State = collections.namedtuple('NTM2State', ('access_output', 'access_state',
                                               'controller_state'))


class NTM2(snt.RNNCore):
  """Reduced NTM core module.

  Contains controller and memory access module.
  """

  def __init__(self,
               access_config,
               controller_config,
               output_size,
               clip_value=None,
               name='ntm2'):
    """Initializes the NTM core.

    Args:
      access_config: dictionary of access module configurations.
      controller_config: dictionary of controller (LSTM) module configurations.
      output_size: output dimension size of core.
      clip_value: clips controller and core output values to between
          `[-clip_value, clip_value]` if specified.
      name: module name (default 'ntm2').

    Raises:
      TypeError: if direct_input_size is not None for any access module other
        than KeyValueMemory.
    """
    super(NTM2, self).__init__(name=name)

    with self._enter_variable_scope():
      self._controller = LSTM(**controller_config)
      self._access = access.MemoryAccess(**access_config)

    self._access_output_size = np.prod(self._access.output_size.as_list())
    self._output_size = output_size
    self._clip_value = clip_value or 0

    self._output_size = tf.TensorShape([output_size])
    self._state_size = NTM2State(
        access_output=self._access_output_size,
        access_state=self._access.state_size,
        controller_state=self._controller.state_size)

  def _clip_if_enabled(self, x):
    if self._clip_value > 0:
      return tf.clip_by_value(x, -self._clip_value, self._clip_value)
    else:
      return x

  def _build(self, inputs, prev_state):
    """Connects the reduced NTM core into the graph.

    Args:
      inputs: Tensor input.
      prev_state: A `NTM2State` tuple containing the fields `access_output`,
          `access_state` and `controller_state`. `access_state` is a 3-D Tensor
          of shape `[batch_size, num_reads, word_size]` containing read words.
          `access_state` is a tuple of the access module's state, and
          `controller_state` is a tuple of controller module's state.

    Returns:
      A tuple `(output, next_state)` where `output` is a tensor and `next_state`
      is a `NTM2State` tuple containing the fields `access_output`,
      `access_state`, and `controller_state`.
    """

    prev_access_output = prev_state.access_output
    prev_access_state = prev_state.access_state
    prev_controller_state = prev_state.controller_state

    batch_flatten = snt.BatchFlatten()
    controller_input = tf.concat(
        [batch_flatten(inputs), batch_flatten(prev_access_output)], 1)

    controller_output, controller_state, gates = self._controller(
        controller_input, prev_controller_state)

    controller_output = self._clip_if_enabled(controller_output)
    controller_state = snt.nest.map(self._clip_if_enabled, controller_state)

    access_output, access_state = self._access(controller_output,
                                               prev_access_state)

    output = tf.concat([controller_output, batch_flatten(access_output)], 1)
    output = snt.Linear(
        output_size=self._output_size.as_list()[0],
        name='output_linear')(output)
    output = self._clip_if_enabled(output)

    return output, NTM2State(
        access_output=access_output,
        access_state=access_state,
        controller_state=controller_state), gates

  def initial_state(self, batch_size, dtype=tf.float32):
    return NTM2State(
        controller_state=self._controller.initial_state(batch_size, dtype),
        access_state=self._access.initial_state(batch_size, dtype),
        access_output=tf.zeros(
            [batch_size] + self._access.output_size.as_list(), dtype))

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size
