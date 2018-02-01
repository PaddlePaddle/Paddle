#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

import sys
import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import paddle.v2.fluid.core as core
import paddle.v2.fluid.framework as framework
import paddle.v2.fluid.layers as pd
from paddle.v2.fluid.executor import Executor
from paddle.v2.fluid.nets import SequenceDecoder

dict_size = 30000
source_dict_dim = target_dict_dim = dict_size
src_dict, trg_dict = paddle.dataset.wmt14.get_dict(dict_size)
hidden_dim = 32
word_dim = 16
IS_SPARSE = True
batch_size = 2
max_length = 8
topk_size = 50
trg_dic_size = 10000
beam_size = 2

decoder_size = hidden_dim

place = core.CPUPlace()


def encoder():
    # encoder
    src_word_id = pd.data(
        name="src_word_id", shape=[1], dtype='int64', lod_level=1)
    src_embedding = pd.embedding(
        input=src_word_id,
        size=[dict_size, word_dim],
        dtype='float32',
        is_sparse=IS_SPARSE,
        param_attr=fluid.ParamAttr(name='vemb'))

    fc1 = pd.fc(input=src_embedding, size=hidden_dim * 4, act='tanh')
    lstm_hidden0, lstm_0 = pd.dynamic_lstm(input=fc1, size=hidden_dim * 4)
    encoder_out = pd.sequence_last_step(input=lstm_hidden0)
    return encoder_out


def decoder_train(context):
    # decoder
    trg_language_word = pd.data(
        name="target_language_word", shape=[1], dtype='int64', lod_level=1)
    trg_embedding = pd.embedding(
        input=trg_language_word,
        size=[dict_size, word_dim],
        dtype='float32',
        is_sparse=IS_SPARSE,
        param_attr=fluid.ParamAttr(name='vemb'))

    rnn = pd.DynamicRNN()
    with rnn.block():
        current_word = rnn.step_input(trg_embedding)
        pre_state = rnn.memory(init=context)
        current_state = pd.fc(input=[current_word, pre_state],
                              size=decoder_size,
                              act='tanh')

        current_score = pd.fc(input=current_state,
                              size=target_dict_dim,
                              act='softmax')
        rnn.update_memory(pre_state, current_state)
        rnn.output(current_score)

    return rnn()


def decoder(context, mode='train'):
    '''
    mode: str
        should be train or decode.
    '''
    Cell = SequenceDecoder.Cell
    InputCell = SequenceDecoder.InputCell
    StateCell = SequenceDecoder.StateCell

    trg_language_word = pd.data(
        name="target_language_word", shape=[1], dtype='int64', lod_level=1)
    # this is only used in train
    trg_embedding = pd.embedding(
        input=trg_language_word,
        size=[dict_size, word_dim],
        dtype='float32',
        is_sparse=IS_SPARSE,
        param_attr=fluid.ParamAttr(name='vemb'))

    init_ids = pd.data(name="init_ids", shape=[1], dtype="int64", lod_level=2)

    item_id_init = {'train': trg_language_word, 'decode': init_ids}
    item_id = InputCell('word_id', item_id_init)

    item_vec = InputCell('word_embs', trg_embedding)

    state0 = StateCell('state', context)

    def state0_updater(seqdec):
        rnn = seqdec.train_rnn
        # current_word = InputCell.get(seqdec.mode, item_id.id)
        pre_state = StateCell.get(mode, state0.id, seqdec.counter
                                  if seqdec.mode == 'decode' else None)
        print 'pre_state', pre_state

        if seqdec.mode == 'train':
            current_word_embedding = InputCell.get('train', 'word_embs')
            print 'current_word_embedding', current_word_embedding
        else:
            current_word_embedding = pd.embedding(input=Cell.get(
                'decode', 'step_input', 'word_embs', counter=seqdec.counter))

        # TODO need to make sure fc in different mode share the same paraemters.
        new_state = pd.fc(input=[current_word_embedding, pre_state],
                          size=decoder_size,
                          act='tanh')

        Cell.add_temp_var(seqdec.mode, 'new_state', new_state)
        return new_state

    def scorer(seqdec):
        new_state = Cell.get_temp_var(seqdec.mode, 'new_state')
        new_score = pd.fc(input=new_state, size=target_dict_dim, act='softmax')
        return new_score

    state0.set_updater(state0_updater)
    seqdec = SequenceDecoder(
        item_id, [state0], scorer, other_step_inputs=[item_vec])

    if mode == 'train':
        res = seqdec.train()
        return res
    # return seqdec.decode(beam_size, max_length, topk_size, end_id=10)


def set_init_lod(data, lod, place):
    res = core.LoDTensor()
    res.set(data, place)
    res.set_lod(lod)
    return res


def to_lodtensor(data, place):
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = core.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res


def train_main():
    context = encoder()
    # rnn_out = decoder(context, 'train')
    rnn_out = decoder_train(context)
    print(framework.default_main_program().block(0))
    sys.exit(0)

    label = pd.data(
        name="target_language_next_word", shape=[1], dtype='int64', lod_level=1)
    cost = pd.cross_entropy(input=rnn_out, label=label)
    avg_cost = pd.mean(x=cost)

    optimizer = fluid.optimizer.Adagrad(learning_rate=1e-4)
    optimizer.minimize(avg_cost)

    train_data = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.train(dict_size), buf_size=1000),
        batch_size=batch_size)

    exe = Executor(place)

    exe.run(framework.default_startup_program())

    batch_id = 0
    for pass_id in xrange(1):
        for data in train_data():
            word_data = to_lodtensor(map(lambda x: x[0], data), place)
            trg_word = to_lodtensor(map(lambda x: x[1], data), place)
            trg_word_next = to_lodtensor(map(lambda x: x[2], data), place)
            outs = exe.run(framework.default_main_program(),
                           feed={
                               'src_word_id': word_data,
                               'target_language_word': trg_word,
                               'target_language_next_word': trg_word_next
                           },
                           fetch_list=[avg_cost])
            avg_cost_val = np.array(outs[0])
            print('pass_id=' + str(pass_id) + ' batch=' + str(batch_id) +
                  " avg_cost=" + str(avg_cost_val))
            if batch_id > 0:
                break
            batch_id += 1


def decode_main():
    context = encoder()
    translation_ids, translation_scores = decoder(context, 'decode')

    exe = Executor(place)
    exe.run(framework.default_startup_program())

    init_ids_data = np.array([1 for _ in range(batch_size)], dtype='int64')
    init_scores_data = np.array(
        [1. for _ in range(batch_size)], dtype='float32')
    init_ids_data = init_ids_data.reshape((batch_size, 1))
    init_scores_data = init_scores_data.reshape((batch_size, 1))
    init_lod = [i for i in range(batch_size)] + [batch_size]
    init_lod = [init_lod, init_lod]

    train_data = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.train(dict_size), buf_size=1000),
        batch_size=batch_size)
    for _, data in enumerate(train_data()):
        init_ids = set_init_lod(init_ids_data, init_lod, place)
        init_scores = set_init_lod(init_scores_data, init_lod, place)

        src_word_data = to_lodtensor(map(lambda x: x[0], data), place)

        result_ids, result_scores = exe.run(
            framework.default_main_program(),
            feed={'init_ids': init_ids,
                  'init_scores': init_scores},
            fetch_list=[translation_ids, translation_scores],
            return_numpy=False)
        print result_ids.lod()
        break


if __name__ == '__main__':
    train_main()
    # decode_main()
