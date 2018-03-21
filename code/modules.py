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

"""
Note: the design of the final model is heavily inspired by the open-source r-net implementation
found at https://github.com/minsangkim142/R-net. When encountering problems in our code, the TA
encouraged us to search for an open-source version to solve the problem. All of the code was only
referenced, and our model below was all hand-coded from our understanding of it. Our painful previous
implementation of the self-attention layer and answer pointer layer available upon request.

Specific inspirations include
- using a separate module to store all params. In addition, we also shrunk the hidden size down
    at all layers since the r-net paper did not provide any dimensions. This helped us with
    out-of-memory issues we were experiencing.
- a wrapper around an RNN cell to attend the input before feeding it into the cell. a large
    headache was figuring out how to attend the input before each time step and this was a 
    beautiful solution to it.
- a separate general function to calculate attention. The attention equations that are used
    four times throughout the r-net paper are all very similar, and making it a separate function
    dramatically improved readability.
"""

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell
import pdb

def Params(hidden_size, question_len):
    """
    All of the parameters used in the network
    """
    with tf.variable_scope("params"):
        h = hidden_size
        # TODO: write comments for each of these, purpose and where used
        return {# for question hidden, in GABR/pointer network
                "W_uQ":tf.get_variable("W_uQ",shape=(2*h, h), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer()),
                # for passage hidden, in GABR
                "W_uP":tf.get_variable("W_uP", shape=(2*h, h), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer()),
                # for question-aware hidden, in self-attention
                "W_vP":tf.get_variable("W_vP", shape=(h, h), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer()),
                # for question-aware hidden, in self-attention
                "W_vPhat":tf.get_variable("W_vPhat", shape=(h, h), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer()),
                # for blend of passage hidden and attention, in GABR
                "W_g":tf.get_variable("W_g", shape=(4*h, 4*h), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer()),
                # for blend of question-aware hidden and attention, in self-attention
                "W_g_self":tf.get_variable("W_g_self", shape=(2*h, 2*h), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer()),
                # for self-attnded passage hidden, in pointer network
                "W_hP":tf.get_variable("W_hP", shape=(2*h, h), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer()),
                # for answer RNN hidden, in pointer network
                "W_ha":tf.get_variable("W_ha", shape=(2*h, h), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer()),
                # for question param, in pointer network init state
                "W_vQ":tf.get_variable("W_vQ", shape=(h, h), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer()),
                # question param, in pointer network init state
                "V_rQ":tf.get_variable("V_rQ", shape=(question_len, h), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer()),
                # attention vector, in GABR, self-attention, pointer network init state, pointer network
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

            if masks is not None:
                input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)
                (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs,
                                                                      input_lens, dtype=tf.float32)
            else:
                (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs,
                                                                      dtype=tf.float32)
            # Note: fw_out and bw_out are the hidden states for every time step.
            # Each is shape (batch_size, seq_len, hidden_size).

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class Attention(object):
    """
    Since attention is calculated as a layer twice (gated and self-attention), we created a single
    layer to represent it, to be initialized with different parameters for each layer.
    """

    def __init__(self, keep_prob, hidden_size):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob

    def build_graph(self, inputs, inputs_mask, memory, memory_mask, params, self_matching):
        """
        Inputs:
          inputs: Tensor shape (batch_size, context_len, input_size)
          inputs_masks: Tensor shape (batch_size, context_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.
          memory: Tensor shape (batch_size, memory_len, input_size)
          memory_mask: Tensor shape (batch_size, memory_len)
          params: weight matrices and other network parameters
          self_matching: boolean, False for GABR, True for self-attention

        Returns:
          outputs: Tensor shape (batch_size, context_len, hidden_size) for GABR.
                   Tensor shape (batch_size, context_len, hidden_size*2) for self-attention.
        """
        with vs.variable_scope("GatedAttn"):
            cell = Attended_Rnn(self.hidden_size, memory, memory_mask, params, self_matching)
            cell = DropoutWrapper(cell, input_keep_prob=self.keep_prob)
            if self_matching:
                cell_bw = Attended_Rnn(self.hidden_size, memory, memory_mask, params, self_matching)
                cell_bw = DropoutWrapper(cell_bw, input_keep_prob=self.keep_prob)
            inputs_masked_len = tf.reduce_sum(inputs_mask, 1)
            if not self_matching:
                # outputs shape (batch_size, context_len, hidden_size)
                outputs, _ = tf.nn.dynamic_rnn(cell, inputs,
                    sequence_length=inputs_masked_len, dtype=tf.float32)
            else:
                (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(cell, cell_bw, inputs,
                    sequence_length=inputs_masked_len, dtype=tf.float32)
                # outputs shape (batch_size, context_len, 2*hidden_size)
                outputs = tf.concat((fw_out, bw_out), 2)
            return outputs


class Attended_Rnn(RNNCell):
    """
    Since both GABR and self-attention have to attend the inputs at each time step of an RNN,
    we created a wrapper around a cell to do so. Inspired by an open-source implementation of
    r-net referenced above.
    """
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
        inputs = attend_inputs(inputs, state, self.memory, self.memory_mask, self.hidden_size, self.params, self.self_matching)
        output, new_state = self.cell(inputs, state, scope)
        return output, new_state


def attend_inputs(inputs, state, memory, memory_mask, hidden_size, params, self_matching):
    """
    Inputs:
      inputs: Tensor shape (batch_size, context_len, input_size)
      state: Tensor shape (batch_size, hidden_size)
      memory: Tensor shape (batch_size, memory_len, input_size)
      memory_mask: Tensor shape (batch_size, memory_len)
      params: weight matrices and other network parameters
      self_matching: boolean, False for GABR, True for self-attention

    Returns:
      outputs: Tensor shape (batch_size, context_len, input_size), the attended inputs
    """
    with tf.variable_scope("attend_inputs"):
        attn_params = params["attn_params"]
        W_g = params["W_g"]
        attn_inputs = [memory, inputs]
        if not self_matching:
            state = tf.reshape(state, (-1, hidden_size))
            attn_inputs.append(state)
        _, attn_dist = attention(attn_params, attn_inputs, memory_mask, hidden_size)
        attn_dist = tf.expand_dims(attn_dist, 2)
        attn_output = tf.reduce_sum(attn_dist * memory, 1) # shape (batch_size, memory_hidden_size)
        new_inputs = tf.concat([inputs, attn_output], 1)
        g = tf.sigmoid(tf.matmul(new_inputs, W_g)) # shape (batch_size, 4*hidden_size)
        return g * new_inputs


def attention(attn_params, inputs, memory_mask, hidden_size):
    """
    Inputs:
      attn_params: map of list of weight matrices and v
      inputs: Tensor shape (batch_size, context_len, input_size)
      memory_mask: Tensor shape (batch_size, memory_len)
      hidden_size: int. Hidden size of the RNN

    Returns:
      masked_logits and prob_dist, Tensors shape (batch_size, memory_len)
    """
    with tf.variable_scope("attention"):
        weights = attn_params["weights"]
        v = attn_params["v"]
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
        return masked_softmax(s, memory_mask, 1)


class AnswerPointer(object):
    """
    Uses the Ptr-Net model to predict both the start and end indices of the answer.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob

    def build_graph(self, self_attn_output, params, question_hiddens, question_mask, context_mask):
        """
        The output layer of r-net, an answer pointer network

        Inputs:
            self_attn_output: Tensor shape (batch_size, context_len, 2*hidden_size)
            params: map of list of weight matrices and v
            question_hiddens: Tensor shape (batch_size, question_len, 2*hidden_size)
            question_mask: Tensor shape (batch_size, question_len)
            context_mask: Tensor shape (batch_size, context_len)

        Outputs:
          logits_start: raw logits before softmax
          probdist_start: probability distribution of the start index
          logits_end: raw logits before softmax
          probdist_end: probability distribution of the end index
        """
        with vs.variable_scope("AnswerPointer"):
            init_attn_params = params["init_attn_params"]
            attn_params = params["attn_params"]
            V_rQ = params["V_rQ"]
            init_attn_inputs = [question_hiddens, V_rQ]
            _, init_attn_dist = attention(init_attn_params, init_attn_inputs, question_mask, self.hidden_size)
            init_attn_dist = tf.expand_dims(init_attn_dist, 2)
            init_state = tf.reduce_sum(init_attn_dist * question_hiddens, 1) # shape (batch_size, 2*hidden_state)
            attn_inputs = [self_attn_output, init_state]
            logits_start, probdist_start = attention(attn_params, attn_inputs, context_mask, self.hidden_size)
            attn_dist = tf.expand_dims(probdist_start, 2)
            rnn_input = tf.reduce_sum(attn_dist * self_attn_output, 1) # shape (batch_size, 2*hidden_state)
            ans_point_rnn = rnn_cell.GRUCell(2*self.hidden_size)
            ans_point_rnn = DropoutWrapper(ans_point_rnn, input_keep_prob=self.keep_prob)
            _, state = tf.nn.static_rnn(ans_point_rnn, [rnn_input], initial_state=init_state, dtype=tf.float32)
            attn_inputs2 = [self_attn_output, state]
            logits_end, probdist_end = attention(attn_params, attn_inputs2, context_mask, self.hidden_size)
            return logits_start, probdist_start, logits_end, probdist_end


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
