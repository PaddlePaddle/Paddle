#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.rnn_cell_impl import RNNCell, BasicLSTMCell
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder
import numpy as np
import os
import argparse
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--embedding_dim",
    type=int,
    default=512,
    help="The dimension of embedding table. (default: %(default)d)")
parser.add_argument(
    "--encoder_size",
    type=int,
    default=512,
    help="The size of encoder bi-rnn unit. (default: %(default)d)")
parser.add_argument(
    "--decoder_size",
    type=int,
    default=512,
    help="The size of decoder rnn unit. (default: %(default)d)")
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    help="The sequence number of a mini-batch data. (default: %(default)d)")
parser.add_argument(
    "--dict_size",
    type=int,
    default=30000,
    help="The dictionary capacity. Dictionaries of source sequence and "
    "target dictionary have same capacity. (default: %(default)d)")
parser.add_argument(
    "--max_time_steps",
    type=int,
    default=81,
    help="Max number of time steps for sequence. (default: %(default)d)")
parser.add_argument(
    "--pass_num",
    type=int,
    default=10,
    help="The pass number to train. (default: %(default)d)")
parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.0002,
    help="Learning rate used to train the model. (default: %(default)f)")
parser.add_argument(
    "--infer_only", action='store_true', help="If set, run forward only.")
parser.add_argument(
    "--beam_size",
    type=int,
    default=3,
    help="The width for beam searching. (default: %(default)d)")
parser.add_argument(
    "--max_generation_length",
    type=int,
    default=250,
    help="The maximum length of sequence when doing generation. "
    "(default: %(default)d)")
parser.add_argument(
    "--save_freq",
    type=int,
    default=500,
    help="Save model checkpoint every this interation. (default: %(default)d)")
parser.add_argument(
    "--model_dir",
    type=str,
    default='./checkpoint',
    help="Path to save model checkpoints. (default: %(default)d)")

_Linear = core_rnn_cell._Linear  # pylint: disable=invalid-name

START_TOKEN_IDX = 0
END_TOKEN_IDX = 1


class LSTMCellWithSimpleAttention(RNNCell):
    """Add attention mechanism to BasicLSTMCell.
    This class is a wrapper based on tensorflow's `BasicLSTMCell`.
    """

    def __init__(self,
                 num_units,
                 encoder_vector,
                 encoder_proj,
                 source_sequence_length,
                 forget_bias=1.0,
                 state_is_tuple=True,
                 activation=None,
                 reuse=None):
        super(LSTMCellWithSimpleAttention, self).__init__(_reuse=reuse)
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will "
                         "soon be deprecated. Use state_is_tuple=True.", self)
        self._num_units = num_units
        # set padding part to 0
        self._encoder_vector = self._reset_padding(encoder_vector,
                                                   source_sequence_length)
        self._encoder_proj = self._reset_padding(encoder_proj,
                                                 source_sequence_length)
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh
        self._linear = None

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units) \
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def zero_state(self, batch_size, dtype):
        state_size = self.state_size
        if hasattr(self, "_last_zero_state"):
            (last_state_size, last_batch_size, last_dtype,
             last_output) = getattr(self, "_last_zero_state")
            if (last_batch_size == batch_size and last_dtype == dtype and
                    last_state_size == state_size):
                return last_output
        with ops.name_scope(
                type(self).__name__ + "ZeroState", values=[batch_size]):
            output = _zero_state_tensors(state_size, batch_size, dtype)
        self._last_zero_state = (state_size, batch_size, dtype, output)
        return output

    def call(self, inputs, state):
        sigmoid = math_ops.sigmoid
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)

        # get context from encoder outputs
        context = self._simple_attention(self._encoder_vector,
                                         self._encoder_proj, h)

        if self._linear is None:
            self._linear = _Linear([inputs, context, h], 4 * self._num_units,
                                   True)
        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = array_ops.split(
            value=self._linear([inputs, context, h]),
            num_or_size_splits=4,
            axis=1)

        new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) *
                 self._activation(j))
        new_h = self._activation(new_c) * sigmoid(o)

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)
        return new_h, new_state

    def _simple_attention(self, encoder_vec, encoder_proj, decoder_state):
        """Implement the attention function.
        The implementation has the same logic to the fluid decoder.
        """
        decoder_state_proj = tf.contrib.layers.fully_connected(
            inputs=decoder_state,
            num_outputs=self._num_units,
            activation_fn=None,
            biases_initializer=None)
        decoder_state_expand = tf.tile(
            tf.expand_dims(
                input=decoder_state_proj, axis=1),
            [1, tf.shape(encoder_proj)[1], 1])
        concated = tf.concat([decoder_state_expand, encoder_proj], axis=2)
        # need reduce the first dimension
        attention_weights = tf.contrib.layers.fully_connected(
            inputs=tf.reshape(
                concated, shape=[-1, self._num_units * 2]),
            num_outputs=1,
            activation_fn=tf.nn.tanh,
            biases_initializer=None)
        attention_weights_reshaped = tf.reshape(
            attention_weights, shape=[tf.shape(encoder_vec)[0], -1, 1])
        # normalize the attention weights using softmax
        attention_weights_normed = tf.nn.softmax(
            attention_weights_reshaped, dim=1)
        scaled = tf.multiply(attention_weights_normed, encoder_vec)
        context = tf.reduce_sum(scaled, axis=1)
        return context

    def _reset_padding(self,
                       memory,
                       memory_sequence_length,
                       check_inner_dims_defined=True):
        """Reset the padding part for encoder inputs.
        This funtion comes from tensorflow's `_prepare_memory` function.
        """
        memory = nest.map_structure(
                lambda m: ops.convert_to_tensor(m, name="memory"), memory)
        if memory_sequence_length is not None:
            memory_sequence_length = ops.convert_to_tensor(
                memory_sequence_length, name="memory_sequence_length")
        if check_inner_dims_defined:

            def _check_dims(m):
                if not m.get_shape()[2:].is_fully_defined():
                    raise ValueError(
                        "Expected memory %s to have fully defined inner dims, "
                        "but saw shape: %s" % (m.name, m.get_shape()))

            nest.map_structure(_check_dims, memory)
        if memory_sequence_length is None:
            seq_len_mask = None
        else:
            seq_len_mask = array_ops.sequence_mask(
                memory_sequence_length,
                maxlen=array_ops.shape(nest.flatten(memory)[0])[1],
                dtype=nest.flatten(memory)[0].dtype)
            seq_len_batch_size = (memory_sequence_length.shape[0].value or
                                  array_ops.shape(memory_sequence_length)[0])

        def _maybe_mask(m, seq_len_mask):
            rank = m.get_shape().ndims
            rank = rank if rank is not None else array_ops.rank(m)
            extra_ones = array_ops.ones(rank - 2, dtype=dtypes.int32)
            m_batch_size = m.shape[0].value or array_ops.shape(m)[0]
            if memory_sequence_length is not None:
                message = ("memory_sequence_length and memory tensor "
                           "batch sizes do not match.")
                with ops.control_dependencies([
                        check_ops.assert_equal(
                            seq_len_batch_size, m_batch_size, message=message)
                ]):
                    seq_len_mask = array_ops.reshape(
                        seq_len_mask,
                        array_ops.concat(
                            (array_ops.shape(seq_len_mask), extra_ones), 0))
                return m * seq_len_mask
            else:
                return m

        return nest.map_structure(lambda m: _maybe_mask(m, seq_len_mask),
                                  memory)


def seq_to_seq_net(embedding_dim, encoder_size, decoder_size, source_dict_dim,
                   target_dict_dim, is_generating, beam_size,
                   max_generation_length):
    src_word_idx = tf.placeholder(tf.int32, shape=[None, None])
    src_sequence_length = tf.placeholder(tf.int32, shape=[None, ])

    src_embedding_weights = tf.get_variable("source_word_embeddings",
                                            [source_dict_dim, embedding_dim])
    src_embedding = tf.nn.embedding_lookup(src_embedding_weights, src_word_idx)

    src_forward_cell = tf.nn.rnn_cell.BasicLSTMCell(encoder_size)
    src_reversed_cell = tf.nn.rnn_cell.BasicLSTMCell(encoder_size)
    # no peephole
    encoder_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=src_forward_cell,
        cell_bw=src_reversed_cell,
        inputs=src_embedding,
        sequence_length=src_sequence_length,
        dtype=tf.float32)

    # concat the forward outputs and backward outputs
    encoded_vec = tf.concat(encoder_outputs, axis=2)

    # project the encoder outputs to size of decoder lstm
    encoded_proj = tf.contrib.layers.fully_connected(
        inputs=tf.reshape(
            encoded_vec, shape=[-1, embedding_dim * 2]),
        num_outputs=decoder_size,
        activation_fn=None,
        biases_initializer=None)
    encoded_proj_reshape = tf.reshape(
        encoded_proj, shape=[-1, tf.shape(encoded_vec)[1], decoder_size])

    # get init state for decoder lstm's H
    backword_first = tf.slice(encoder_outputs[1], [0, 0, 0], [-1, 1, -1])
    decoder_boot = tf.contrib.layers.fully_connected(
        inputs=tf.reshape(
            backword_first, shape=[-1, embedding_dim]),
        num_outputs=decoder_size,
        activation_fn=tf.nn.tanh,
        biases_initializer=None)

    # prepare the initial state for decoder lstm
    cell_init = tf.zeros(tf.shape(decoder_boot), tf.float32)
    initial_state = LSTMStateTuple(cell_init, decoder_boot)

    # create decoder lstm cell
    decoder_cell = LSTMCellWithSimpleAttention(
        decoder_size,
        encoded_vec
        if not is_generating else seq2seq.tile_batch(encoded_vec, beam_size),
        encoded_proj_reshape if not is_generating else
        seq2seq.tile_batch(encoded_proj_reshape, beam_size),
        src_sequence_length if not is_generating else
        seq2seq.tile_batch(src_sequence_length, beam_size),
        forget_bias=0.0)

    output_layer = Dense(target_dict_dim, name='output_projection')

    if not is_generating:
        trg_word_idx = tf.placeholder(tf.int32, shape=[None, None])
        trg_sequence_length = tf.placeholder(tf.int32, shape=[None, ])
        trg_embedding_weights = tf.get_variable(
            "target_word_embeddings", [target_dict_dim, embedding_dim])
        trg_embedding = tf.nn.embedding_lookup(trg_embedding_weights,
                                               trg_word_idx)

        training_helper = seq2seq.TrainingHelper(
            inputs=trg_embedding,
            sequence_length=trg_sequence_length,
            time_major=False,
            name='training_helper')

        training_decoder = seq2seq.BasicDecoder(
            cell=decoder_cell,
            helper=training_helper,
            initial_state=initial_state,
            output_layer=output_layer)

        # get the max length of target sequence
        max_decoder_length = tf.reduce_max(trg_sequence_length)

        decoder_outputs_train, _, _ = seq2seq.dynamic_decode(
            decoder=training_decoder,
            output_time_major=False,
            impute_finished=True,
            maximum_iterations=max_decoder_length)

        decoder_logits_train = tf.identity(decoder_outputs_train.rnn_output)
        decoder_pred_train = tf.argmax(
            decoder_logits_train, axis=-1, name='decoder_pred_train')
        masks = tf.sequence_mask(
            lengths=trg_sequence_length,
            maxlen=max_decoder_length,
            dtype=tf.float32,
            name='masks')

        # place holder of label sequence
        lbl_word_idx = tf.placeholder(tf.int32, shape=[None, None])

        # compute the loss
        loss = seq2seq.sequence_loss(
            logits=decoder_logits_train,
            targets=lbl_word_idx,
            weights=masks,
            average_across_timesteps=True,
            average_across_batch=True)

        # return feeding list and loss operator
        return {
            'src_word_idx': src_word_idx,
            'src_sequence_length': src_sequence_length,
            'trg_word_idx': trg_word_idx,
            'trg_sequence_length': trg_sequence_length,
            'lbl_word_idx': lbl_word_idx
        }, loss
    else:
        start_tokens = tf.ones([tf.shape(src_word_idx)[0], ],
                               tf.int32) * START_TOKEN_IDX
        # share the same embedding weights with target word
        trg_embedding_weights = tf.get_variable(
            "target_word_embeddings", [target_dict_dim, embedding_dim])

        inference_decoder = beam_search_decoder.BeamSearchDecoder(
            cell=decoder_cell,
            embedding=lambda tokens: tf.nn.embedding_lookup(trg_embedding_weights, tokens),
            start_tokens=start_tokens,
            end_token=END_TOKEN_IDX,
            initial_state=tf.nn.rnn_cell.LSTMStateTuple(
                tf.contrib.seq2seq.tile_batch(initial_state[0], beam_size),
                tf.contrib.seq2seq.tile_batch(initial_state[1], beam_size)),
            beam_width=beam_size,
            output_layer=output_layer)

        decoder_outputs_decode, _, _ = seq2seq.dynamic_decode(
            decoder=inference_decoder,
            output_time_major=False,
            #impute_finished=True,# error occurs
            maximum_iterations=max_generation_length)

        predicted_ids = decoder_outputs_decode.predicted_ids

        return {
            'src_word_idx': src_word_idx,
            'src_sequence_length': src_sequence_length
        }, predicted_ids


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in vars(args).iteritems():
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def padding_data(data, padding_size, value):
    data = data + [value] * padding_size
    return data[:padding_size]


def save(sess, path, var_list=None, global_step=None):
    saver = tf.train.Saver(var_list)
    save_path = saver.save(sess, save_path=path, global_step=global_step)
    print('Model save at %s' % save_path)


def restore(sess, path, var_list=None):
    # var_list = None returns the list of all saveable variables
    saver = tf.train.Saver(var_list)
    saver.restore(sess, save_path=path)
    print('model restored from %s' % path)


def adapt_batch_data(data):
    src_seq = map(lambda x: x[0], data)
    trg_seq = map(lambda x: x[1], data)
    lbl_seq = map(lambda x: x[2], data)

    src_sequence_length = np.array(
        [len(seq) for seq in src_seq]).astype('int32')
    src_seq_maxlen = np.max(src_sequence_length)

    trg_sequence_length = np.array(
        [len(seq) for seq in trg_seq]).astype('int32')
    trg_seq_maxlen = np.max(trg_sequence_length)

    src_seq = np.array(
        [padding_data(seq, src_seq_maxlen, END_TOKEN_IDX)
         for seq in src_seq]).astype('int32')

    trg_seq = np.array(
        [padding_data(seq, trg_seq_maxlen, END_TOKEN_IDX)
         for seq in trg_seq]).astype('int32')

    lbl_seq = np.array(
        [padding_data(seq, trg_seq_maxlen, END_TOKEN_IDX)
         for seq in lbl_seq]).astype('int32')

    return {
        'src_word_idx': src_seq,
        'src_sequence_length': src_sequence_length,
        'trg_word_idx': trg_seq,
        'trg_sequence_length': trg_sequence_length,
        'lbl_word_idx': lbl_seq
    }


def train():
    feeding_dict, loss = seq_to_seq_net(
        embedding_dim=args.embedding_dim,
        encoder_size=args.encoder_size,
        decoder_size=args.decoder_size,
        source_dict_dim=args.dict_size,
        target_dict_dim=args.dict_size,
        is_generating=False,
        beam_size=args.beam_size,
        max_generation_length=args.max_generation_length)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    trainable_params = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

    gradients = tf.gradients(loss, trainable_params)
    # may clip the parameters
    clip_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

    updates = optimizer.apply_gradients(
        zip(gradients, trainable_params), global_step=global_step)

    src_dict, trg_dict = paddle.dataset.wmt14.get_dict(args.dict_size)

    train_batch_generator = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.train(args.dict_size), buf_size=1000),
        batch_size=args.batch_size)

    test_batch_generator = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.test(args.dict_size), buf_size=1000),
        batch_size=args.batch_size)

    def do_validataion():
        total_loss = 0.0
        count = 0
        for batch_id, data in enumerate(test_batch_generator()):
            adapted_batch_data = adapt_batch_data(data)
            outputs = sess.run([loss],
                               feed_dict={
                                   item[1]: adapted_batch_data[item[0]]
                                   for item in feeding_dict.items()
                               })
            total_loss += outputs[0]
            count += 1
        return total_loss / count

    config = tf.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        sess.run(init_l)
        sess.run(init_g)
        for pass_id in xrange(args.pass_num):
            pass_start_time = time.time()
            words_seen = 0
            for batch_id, data in enumerate(train_batch_generator()):
                adapted_batch_data = adapt_batch_data(data)
                words_seen += np.sum(adapted_batch_data['src_sequence_length'])
                words_seen += np.sum(adapted_batch_data['trg_sequence_length'])
                outputs = sess.run([updates, loss],
                                   feed_dict={
                                       item[1]: adapted_batch_data[item[0]]
                                       for item in feeding_dict.items()
                                   })
                print("pass_id=%d, batch_id=%d, train_loss: %f" %
                      (pass_id, batch_id, outputs[1]))
            pass_end_time = time.time()
            test_loss = do_validataion()
            time_consumed = pass_end_time - pass_start_time
            words_per_sec = words_seen / time_consumed
            print("pass_id=%d, test_loss: %f, words/s: %f, sec/pass: %f" %
                  (pass_id, test_loss, words_per_sec, time_consumed))


def infer():
    feeding_dict, predicted_ids = seq_to_seq_net(
        embedding_dim=args.embedding_dim,
        encoder_size=args.encoder_size,
        decoder_size=args.decoder_size,
        source_dict_dim=args.dict_size,
        target_dict_dim=args.dict_size,
        is_generating=True,
        beam_size=args.beam_size,
        max_generation_length=args.max_generation_length)

    src_dict, trg_dict = paddle.dataset.wmt14.get_dict(args.dict_size)
    test_batch_generator = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.train(args.dict_size), buf_size=1000),
        batch_size=args.batch_size)

    config = tf.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    with tf.Session(config=config) as sess:
        restore(sess, './checkpoint/tf_seq2seq-1500')
        for batch_id, data in enumerate(test_batch_generator()):
            src_seq = map(lambda x: x[0], data)

            source_language_seq = [
                src_dict[item] for seq in src_seq for item in seq
            ]

            src_sequence_length = np.array(
                [len(seq) for seq in src_seq]).astype('int32')
            src_seq_maxlen = np.max(src_sequence_length)
            src_seq = np.array([
                padding_data(seq, src_seq_maxlen, END_TOKEN_IDX)
                for seq in src_seq
            ]).astype('int32')

            outputs = sess.run([predicted_ids],
                               feed_dict={
                                   feeding_dict['src_word_idx']: src_seq,
                                   feeding_dict['src_sequence_length']:
                                   src_sequence_length
                               })

            print("\nDecoder result comparison: ")
            source_language_seq = ' '.join(source_language_seq).lstrip(
                '<s>').rstrip('<e>').strip()
            inference_seq = ''
            print(" --> source: " + source_language_seq)
            for item in outputs[0][0]:
                if item[0] == END_TOKEN_IDX: break
                inference_seq += ' ' + trg_dict.get(item[0], '<unk>')
            print(" --> inference: " + inference_seq)


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)
    if args.infer_only:
        infer()
    else:
        train()
