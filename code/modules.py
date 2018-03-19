# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


def Params(hidden_size):
    """
    All of the parameters used in the network
    """
    with tf.variable_scope("params"):
        h = hidden_size
        return {"W_uQ":tf.get_variable("W_uQ",shape=(2*h, h), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer()),
                "W_uP":tf.get_variable("W_uP", shape=(2*h, h), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer()),
                "W_vP":tf.get_variable("W_vP", shape=(h, h), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer()),
                "W_vP2":tf.get_variable("W_vP2", shape=(2*h, h), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer()),
                "W_g":tf.get_variable("W_g", shape=(4*h, 4*h), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer()),
                "W_g2":tf.get_variable("W_g2", shape=(3*h, 3*h), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer()),
                "W_hP":tf.get_variable("W_hP", shape=(2*h, h), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer()),
                "W_vPhat":tf.get_variable("W_vPhat", shape=(2*h, h), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer()),
                "W_ha":tf.get_variable("W_ha", shape=(2*h, h), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer()),
                "W_vQ":tf.get_variable("W_vQ", shape=(h, h), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer()),
                "v":tf.get_variable("v", shape=(h), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())}


class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


class Attention(object):
    """
    Since attention is calculated as a layer twice (gated and self-attention), we created a single
    layer to represent it, to be initialized with different parameters for each layer.
    """

    def __init__(self, keep_prob, hidden_size):
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob

    def build_graph(self, inputs, inputs_mask, memory, memory_mask, weights, self_matching):
        with vs.variable_scope("GatedAttn"):
            cell = Attended_Rnn(self.hidden_size, memory, memory_mask, weights, self_matching)
            cell = DropoutWrapper(cell, input_keep_prob=self.keep_prob)
            inputs_masked_len = tf.reduce_sum(inputs_mask, 1)
            # outputs shape (batch_size, context_len, hidden_size)
            outputs, _ = tf.nn.dynamic_rnn(cell, inputs, sequence_length=inputs_masked_len, dtype=tf.float32)
            return outputs


class Attended_Rnn(RNNCell):
  def __init__(self, hidden_size, memory, memory_mask, params, self_matching):
    super(RNNCell, self).__init__()
    self.cell = rnn_cell.GRUCell(hidden_size)
    self.hidden_size = hidden_size
    self.memory = memory
    self.memory_mask = memory_mask
    self.params = params
    self.self_matching = self_matching

  @property
  def state_size(self):
    return self.hidden_size

  @property
  def output_size(self):
    return self.hidden_size

  def __call__(self, inputs, state, scope = None):
    """Gated recurrent unit (GRU) with nunits cells."""
    inputs = attend_inputs(inputs, state, self.memory, self.memory_mask, self.hidden_size, self.params, self.self_matching)
    output, new_state = self.cell(inputs, state, scope)
    return output, new_state


def attend_inputs(inputs, state, memory, memory_mask, hidden_size, params, self_matching):
    """
    inputs: shape (batch_size, 2*hidden_size) if gated, (batch_size, hidden_size) if self
    state: shape (batch_size, hidden_size)
    memory: shape (batch_size, question_len, 2*hidden_size)
    """
    with tf.variable_scope("attend_inputs"):
        params, W_g = params
        attn_inputs = [memory, inputs]
        if not self_matching:
            state = tf.reshape(state, (-1, hidden_size))
            attn_inputs.append(state)
        attn_dist = attention(params, attn_inputs, memory_mask, hidden_size)
        attn_dist = tf.expand_dims(attn_dist, 2)
        attn_output = tf.reduce_sum(attn_dist * memory, 1) # shape (batch_size, 2*hidden_size)
        new_inputs = tf.concat([inputs, attn_output], 1)
        g = tf.sigmoid(tf.matmul(new_inputs, W_g)) # shape (batch_size, 4*hidden_size)
        return g * new_inputs


def attention(params, inputs, memory_mask, hidden_size):
    """
    TODO: update
    inputs: shape (batch_size, 2*hidden_size)
    state: shape (batch_size, hidden_size)
    memory: shape (batch_size, question_len, 2*hidden_size)
    wuq 2hxh
    wup hxh
    wvp 2hxh

    """
    with tf.variable_scope("attention"):
        weights, v = params
        results = []
        for weight, ainput in zip(weights, inputs):
            input_shape = ainput.shape.as_list()
            ainput = tf.reshape(ainput, (-1, input_shape[-1]))
            result = tf.matmul(ainput, weight)
            if len(input_shape) > 2:
                result = tf.reshape(result, (-1, input_shape[1], hidden_size))
            elif input_shape[0] is None:
                result = tf.expand_dims(result, 1)
            else:
                result = tf.expand_dims(result, 0)
            results.append(result)
        s = tf.reduce_sum(tf.tanh(sum(results)) * v, [-1]) # shape (batch_size, memory_len)
        _, attn_dist = masked_softmax(s, memory_mask, 1)
        return attn_dist


def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist
