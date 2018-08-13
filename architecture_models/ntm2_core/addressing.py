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
"""NTM addressing modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import sonnet as snt
import tensorflow as tf

import util

# Ensure values are greater than epsilon to avoid numerical instability.
_EPSILON = 1e-6

def _vector_norms(m):
  squared_norms = tf.reduce_sum(m * m, axis=2, keep_dims=True)
  return tf.sqrt(squared_norms + _EPSILON)


def weighted_softmax(activations, strengths, strengths_op):
  """Returns softmax over activations multiplied by positive strengths.

  Args:
    activations: A tensor of shape `[batch_size, num_heads, memory_size]`, of
      activations to be transformed. Softmax is taken over the last dimension.
    strengths: A tensor of shape `[batch_size, num_heads]` containing strengths to
      multiply by the activations prior to the softmax.
    strengths_op: An operation to transform strengths before softmax.

  Returns:
    A tensor of same shape as `activations` with weighted softmax applied.
  """
  transformed_strengths = tf.expand_dims(strengths_op(strengths), -1)
  sharp_activations = activations * transformed_strengths
  softmax = snt.BatchApply(module_or_op=tf.nn.softmax)
  return softmax(sharp_activations)


class CosineWeights(snt.AbstractModule):
  """Cosine-weighted attention.

  Calculates the cosine similarity between a query and each word in memory, then
  applies a weighted softmax to return a sharp distribution.
  """

  def __init__(self,
               num_heads,
               word_size,
               strength_op=tf.nn.softplus,
               name='cosine_weights'):
    """Initializes the CosineWeights module.

    Args:
      num_heads: number of memory heads.
      word_size: memory word size.
      strength_op: operation to apply to strengths (default is tf.nn.softplus).
      name: module name (default 'cosine_weights')
    """
    super(CosineWeights, self).__init__(name=name)
    self._num_heads = num_heads
    self._word_size = word_size
    self._strength_op = strength_op

  def _build(self, memory, keys, strengths):
    """Connects the CosineWeights module into the graph.

    Args:
      memory: A 3-D tensor of shape `[batch_size, memory_size, word_size]`.
      keys: A 3-D tensor of shape `[batch_size, num_heads, word_size]`.
      strengths: A 2-D tensor of shape `[batch_size, num_heads]`.

    Returns:
      Weights tensor of shape `[batch_size, num_heads, memory_size]`.
    """
    # Calculates the inner product between the query vector and words in memory.
    dot = tf.matmul(keys, memory, adjoint_b=True)

    # Outer product to compute denominator (euclidean norm of query and memory).
    memory_norms = _vector_norms(memory)
    key_norms = _vector_norms(keys)
    norm = tf.matmul(key_norms, memory_norms, adjoint_b=True)

    # Calculates cosine similarity between the query vector and words in memory.
    similarity = dot / (norm + _EPSILON)

    return weighted_softmax(similarity, strengths, self._strength_op)
