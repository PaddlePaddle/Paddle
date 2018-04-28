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
"""seq2seq model for fluid."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import time
import distutils.util

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.framework as framework
from paddle.fluid.executor import Executor

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
    default=16,
    help="The sequence number of a mini-batch data. (default: %(default)d)")
parser.add_argument(
    '--skip_batch_num',
    type=int,
    default=5,
    help='The first num of minibatch num to skip, for better performance test')
parser.add_argument(
    '--iterations', type=int, default=80, help='The number of minibatches.')
parser.add_argument(
    "--dict_size",
    type=int,
    default=30000,
    help="The dictionary capacity. Dictionaries of source sequence and "
    "target dictionary have same capacity. (default: %(default)d)")
parser.add_argument(
    "--pass_num",
    type=int,
    default=2,
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
    '--device',
    type=str,
    default='GPU',
    choices=['CPU', 'GPU'],
    help="The device type.")
parser.add_argument(
    "--max_length",
    type=int,
    default=250,
    help="The maximum length of sequence when doing generation. "
    "(default: %(default)d)")
parser.add_argument(
    '--with_test',
    action='store_true',
    help='If set, test the testset during training.')


def lstm_step(x_t, hidden_t_prev, cell_t_prev, size):
    def linear(inputs):
        return fluid.layers.fc(input=inputs, size=size, bias_attr=True)

    forget_gate = fluid.layers.sigmoid(x=linear([hidden_t_prev, x_t]))
    input_gate = fluid.layers.sigmoid(x=linear([hidden_t_prev, x_t]))
    output_gate = fluid.layers.sigmoid(x=linear([hidden_t_prev, x_t]))
    cell_tilde = fluid.layers.tanh(x=linear([hidden_t_prev, x_t]))

    cell_t = fluid.layers.sums(input=[
        fluid.layers.elementwise_mul(
            x=forget_gate, y=cell_t_prev), fluid.layers.elementwise_mul(
                x=input_gate, y=cell_tilde)
    ])

    hidden_t = fluid.layers.elementwise_mul(
        x=output_gate, y=fluid.layers.tanh(x=cell_t))

    return hidden_t, cell_t


def seq_to_seq_net(embedding_dim, encoder_size, decoder_size, source_dict_dim,
                   target_dict_dim, is_generating, beam_size, max_length):
    """Construct a seq2seq network."""

    def bi_lstm_encoder(input_seq, gate_size):
        # Linear transformation part for input gate, output gate, forget gate
        # and cell activation vectors need be done outside of dynamic_lstm.
        # So the output size is 4 times of gate_size.
        input_forward_proj = fluid.layers.fc(input=input_seq,
                                             size=gate_size * 4,
                                             act=None,
                                             bias_attr=False)
        forward, _ = fluid.layers.dynamic_lstm(
            input=input_forward_proj, size=gate_size * 4, use_peepholes=False)
        input_reversed_proj = fluid.layers.fc(input=input_seq,
                                              size=gate_size * 4,
                                              act=None,
                                              bias_attr=False)
        reversed, _ = fluid.layers.dynamic_lstm(
            input=input_reversed_proj,
            size=gate_size * 4,
            is_reverse=True,
            use_peepholes=False)
        return forward, reversed

    src_word_idx = fluid.layers.data(
        name='source_sequence', shape=[1], dtype='int64', lod_level=1)

    src_embedding = fluid.layers.embedding(
        input=src_word_idx,
        size=[source_dict_dim, embedding_dim],
        dtype='float32')

    src_forward, src_reversed = bi_lstm_encoder(
        input_seq=src_embedding, gate_size=encoder_size)

    encoded_vector = fluid.layers.concat(
        input=[src_forward, src_reversed], axis=1)

    encoded_proj = fluid.layers.fc(input=encoded_vector,
                                   size=decoder_size,
                                   bias_attr=False)

    backward_first = fluid.layers.sequence_pool(
        input=src_reversed, pool_type='first')

    decoder_boot = fluid.layers.fc(input=backward_first,
                                   size=decoder_size,
                                   bias_attr=False,
                                   act='tanh')

    def lstm_decoder_with_attention(target_embedding, encoder_vec, encoder_proj,
                                    decoder_boot, decoder_size):
        def simple_attention(encoder_vec, encoder_proj, decoder_state):
            decoder_state_proj = fluid.layers.fc(input=decoder_state,
                                                 size=decoder_size,
                                                 bias_attr=False)
            decoder_state_expand = fluid.layers.sequence_expand(
                x=decoder_state_proj, y=encoder_proj)
            concated = fluid.layers.concat(
                input=[encoder_proj, decoder_state_expand], axis=1)
            attention_weights = fluid.layers.fc(input=concated,
                                                size=1,
                                                act='tanh',
                                                bias_attr=False)
            attention_weights = fluid.layers.sequence_softmax(
                input=attention_weights)
            weigths_reshape = fluid.layers.reshape(
                x=attention_weights, shape=[-1])
            scaled = fluid.layers.elementwise_mul(
                x=encoder_vec, y=weigths_reshape, axis=0)
            context = fluid.layers.sequence_pool(input=scaled, pool_type='sum')
            return context

        rnn = fluid.layers.DynamicRNN()

        cell_init = fluid.layers.fill_constant_batch_size_like(
            input=decoder_boot,
            value=0.0,
            shape=[-1, decoder_size],
            dtype='float32')
        cell_init.stop_gradient = False

        with rnn.block():
            current_word = rnn.step_input(target_embedding)
            encoder_vec = rnn.static_input(encoder_vec)
            encoder_proj = rnn.static_input(encoder_proj)
            hidden_mem = rnn.memory(init=decoder_boot, need_reorder=True)
            cell_mem = rnn.memory(init=cell_init)
            context = simple_attention(encoder_vec, encoder_proj, hidden_mem)
            decoder_inputs = fluid.layers.concat(
                input=[context, current_word], axis=1)
            h, c = lstm_step(decoder_inputs, hidden_mem, cell_mem, decoder_size)
            rnn.update_memory(hidden_mem, h)
            rnn.update_memory(cell_mem, c)
            out = fluid.layers.fc(input=h,
                                  size=target_dict_dim,
                                  bias_attr=True,
                                  act='softmax')
            rnn.output(out)
        return rnn()

    if not is_generating:
        trg_word_idx = fluid.layers.data(
            name='target_sequence', shape=[1], dtype='int64', lod_level=1)

        trg_embedding = fluid.layers.embedding(
            input=trg_word_idx,
            size=[target_dict_dim, embedding_dim],
            dtype='float32')

        prediction = lstm_decoder_with_attention(trg_embedding, encoded_vector,
                                                 encoded_proj, decoder_boot,
                                                 decoder_size)
        label = fluid.layers.data(
            name='label_sequence', shape=[1], dtype='int64', lod_level=1)
        cost = fluid.layers.cross_entropy(input=prediction, label=label)
        avg_cost = fluid.layers.mean(x=cost)

        feeding_list = ["source_sequence", "target_sequence", "label_sequence"]

        return avg_cost, feeding_list


def to_lodtensor(data, place):
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    lod_t = core.LoDTensor()
    lod_t.set(flattened_data, place)
    lod_t.set_lod([lod])
    return lod_t, lod[-1]


def lodtensor_to_ndarray(lod_tensor):
    dims = lod_tensor.get_dims()
    ndarray = np.zeros(shape=dims).astype('float32')
    for i in xrange(np.product(dims)):
        ndarray.ravel()[i] = lod_tensor.get_float_element(i)
    return ndarray


def train():
    avg_cost, feeding_list = seq_to_seq_net(
        args.embedding_dim,
        args.encoder_size,
        args.decoder_size,
        args.dict_size,
        args.dict_size,
        False,
        beam_size=args.beam_size,
        max_length=args.max_length)

    # clone from default main program
    inference_program = fluid.default_main_program().clone()

    optimizer = fluid.optimizer.Adam(learning_rate=args.learning_rate)
    optimizer.minimize(avg_cost)

    fluid.memory_optimize(fluid.default_main_program())

    train_batch_generator = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.train(args.dict_size), buf_size=1000),
        batch_size=args.batch_size)

    test_batch_generator = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.test(args.dict_size), buf_size=1000),
        batch_size=args.batch_size)

    place = core.CPUPlace() if args.device == 'CPU' else core.CUDAPlace(0)
    exe = Executor(place)
    exe.run(framework.default_startup_program())

    def do_validation():
        total_loss = 0.0
        count = 0
        for batch_id, data in enumerate(test_batch_generator()):
            src_seq = to_lodtensor(map(lambda x: x[0], data), place)[0]
            trg_seq = to_lodtensor(map(lambda x: x[1], data), place)[0]
            lbl_seq = to_lodtensor(map(lambda x: x[2], data), place)[0]

            fetch_outs = exe.run(inference_program,
                                 feed={
                                     feeding_list[0]: src_seq,
                                     feeding_list[1]: trg_seq,
                                     feeding_list[2]: lbl_seq
                                 },
                                 fetch_list=[avg_cost],
                                 return_numpy=False)

            total_loss += lodtensor_to_ndarray(fetch_outs[0])[0]
            count += 1

        return total_loss / count

    iters, num_samples, start_time = 0, 0, time.time()
    for pass_id in xrange(args.pass_num):
        train_accs = []
        train_losses = []
        for batch_id, data in enumerate(train_batch_generator()):
            if iters == args.skip_batch_num:
                start_time = time.time()
                num_samples = 0
            if iters == args.iterations:
                break
            src_seq, word_num = to_lodtensor(map(lambda x: x[0], data), place)
            num_samples += word_num
            trg_seq, word_num = to_lodtensor(map(lambda x: x[1], data), place)
            num_samples += word_num
            lbl_seq, _ = to_lodtensor(map(lambda x: x[2], data), place)

            fetch_outs = exe.run(framework.default_main_program(),
                                 feed={
                                     feeding_list[0]: src_seq,
                                     feeding_list[1]: trg_seq,
                                     feeding_list[2]: lbl_seq
                                 },
                                 fetch_list=[avg_cost])

            iters += 1
            loss = np.array(fetch_outs[0])
            print(
                "Pass = %d, Iter = %d, Loss = %f" % (pass_id, iters, loss)
            )  # The accuracy is the accumulation of batches, but not the current batch.

        train_elapsed = time.time() - start_time
        examples_per_sec = num_samples / train_elapsed
        print('\nTotal examples: %d, total time: %.5f, %.5f examples/sed\n' %
              (num_samples, train_elapsed, examples_per_sec))
        # evaluation
        if args.with_test:
            test_loss = do_validation()
        exit(0)


def infer():
    pass


def print_arguments(args):
    print('----------- seq2seq Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)
    if args.infer_only:
        infer()
    else:
        train()
