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
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell
import pdb

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


class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
                #think about using multiplicative attention
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output


class RNNAttn(object):

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell = DropoutWrapper(self.rnn_cell, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size*4)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*4).

        """
        with vs.variable_scope("RNN_attn"):

            input_lens = tf.reduce_sum(masks, reduction_indices=1)  # shape (batch_size)
            # Note: fw_out are the hidden states for every time step.
            # Each is shape (batch_size, seq_len, hidden_size*4).
            fw_out, _ = tf.nn.dynamic_rnn(self.rnn_cell, inputs, input_lens, dtype=tf.float32)

            # Apply dropout
            out = tf.nn.dropout(fw_out, self.keep_prob)

            return out


class SelfAttn(object):

    def __init__(self, keep_prob, vec_size, context_len):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
        """
        self.keep_prob = keep_prob
        self.vec_size = vec_size
        self.context_len = context_len
        # vector size here is hidden_size*4

    def build_graph(self, reps, mask):
        """
        representations attend to themselves.
        For each representation, return an attention distribution and an attention output vector.

        Inputs:
          inputs: Tensor shape (batch_size, context_len, hidden_size*4)
          masks: Tensor shape (batch_size, context_len).
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          attn_dist: Tensor shape (batch_size, context_len, context_len).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.

          output: Tensor shape Tensor shape (batch_size, context_len, hidden_size*4).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).

        """
        with vs.variable_scope("SelfAttn"):

            W_1 = tf.get_variable("W_1", shape = [self.vec_size, self.vec_size],
                                  dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            W_2 = tf.get_variable("W_2", shape= [self.vec_size, self.vec_size],
                                  dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            # W_1, W_2 shape (hidden_size*4, hidden_size*4)

            v = tf.get_variable("v", shape= [self.vec_size, 1],
                                dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            # v vector shape (hidden_size*4)

            # Calculate self attention distribution
            reps = tf.reshape(reps, [-1, self.vec_size])
            vw1 = tf.matmul(reps, W_1)
            vw1 = tf.reshape(vw1, [-1, self.context_len, self.vec_size])
            # shape (batch_size, context_len, hidden_size*4)

            vw2 = tf.matmul(reps, W_2)
            vw2 = tf.reshape(vw2, [-1, self.context_len, self.vec_size])
            # shape (batch_size, context_len, hidden_size*4)

            sf_attn_logits = []
            # shape would be (batch_size, context_len, context_len) when completed

            for i in xrange(self.context_len):
                # pdb.set_trace()

                vivj = tf.reshape(tf.tanh(vw1[:, i:(i+1), :] + vw2), [-1, self.vec_size])
                # vivj is shape (batch_size*context_len, hidden_size*4)

                temp = tf.matmul(vivj, v)
                temp = tf.reshape(temp, [-1, self.context_len])
                # (batch_size, context_len)

                sf_attn_logits.append(temp)

            sf_attn_logits = tf.stack(sf_attn_logits, axis=1)

            sf_attn_logits_mask = tf.expand_dims(mask, 1)
            # shape (batch_size, 1, context_len)

            _, sf_attn_dist = masked_softmax(sf_attn_logits, sf_attn_logits_mask, 2)
            # shape (batch_size, context_len, context_len). take softmax over

            # Use attention distribution to take weighted sum of values
            reps = tf.reshape(reps, [-1, self.context_len, self.vec_size])
            output = tf.matmul(sf_attn_dist, reps) # shape (batch_size, context_len, hidden_size*4)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return sf_attn_dist, output


class RNNEncoder2(object):
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
        with vs.variable_scope("RNNEncoder2"):

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


# class CharCNN(object):
#     """
#     A character-level CNN for input layer
#
#     module to encode a character-based sequence using a CNN.
#     It feeds the input through a CNN and returns all the hidden states.
#     """
#
#     def __init__(self, keep_prob):
#         """
#         Inputs:
#           hidden_size: int. Hidden size of the CNN
#           keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
#         """
#         self.hidden_size = 100
#         self.keep_prob = keep_prob
#
#     def build_graph(self, inputs):
#
#         with vs.variable_scope("charCNN"):
#
#             filter_shape = [num_quantized_chars, filter_sizes[0], 1, num_filters_per_size]
#             W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
#             b = tf.Variable(tf.constant(0.1, shape=[num_filters_per_size]), name="b")
#             conv = tf.nn.conv2d(self.input_x, W, strides=[1, 1, 1, 1], padding="VALID", name="conv1")
#             h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
#             pooled = tf.nn.max_pool(
#                 h,
#                 ksize=[1, 1, 3, 1],
#                 strides=[1, 1, 3, 1],
#                 padding='VALID',
#                 name="pool1")
#
#         num_features_total = 34 * num_filters_per_size
#         h_pool_flat = tf.reshape(pooled, [-1, num_features_total])
#
#         # Add dropout
#         out = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
#
#         return out