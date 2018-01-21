#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import paddle.v2.fluid.core as core
import paddle.v2.fluid.framework as framework
import paddle.v2.fluid.layers as pd
from paddle.v2.fluid.executor import Executor

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
        mem = rnn.memory(init=context)
        fc1 = pd.fc(input=[current_word, mem], size=decoder_size, act='tanh')
        out = pd.fc(input=fc1, size=target_dict_dim, act='softmax')
        rnn.update_memory(mem, fc1)
        rnn.output(out)

    return rnn()


def decoder_decode(context):
    init_state = context
    array_len = pd.fill_constant(shape=[1], dtype='int64', value=max_length)
    counter = pd.zeros(shape=[1], dtype='int64')

    # fill the first element with init_state
    state_array = pd.create_array('float32')
    pd.array_write(init_state, array=state_array, i=counter)

    # ids, scores as memory
    ids_array = pd.create_array('int64')
    scores_array = pd.create_array('float32')

    init_ids = pd.data(name="init_ids", shape=[1], dtype="int64", lod_level=2)
    init_scores = pd.data(
        name="init_scores", shape=[1], dtype="float32", lod_level=2)

    # init_ids = pd.ones(shape=[batch_size, 1], dtype='int64')
    # init_scores = pd.ones(shape=[batch_size, 1], dtype='float32')
    # init ids to [1..]
    # init scores to [1.]
    pd.array_write(init_ids, array=ids_array, i=counter)
    pd.array_write(init_scores, array=scores_array, i=counter)

    cond = pd.less_than(x=counter, y=array_len)

    while_op = pd.While(cond=cond)
    with while_op.block():
        pre_ids = pd.array_read(array=ids_array, i=counter)
        pre_state = pd.array_read(array=state_array, i=counter)
        pre_score = pd.array_read(array=scores_array, i=counter)

        # pre_state_expanded = pd.sequence_expand(pre_state, pre_score)

        pre_ids_emb = pd.embedding(
            input=pre_ids,
            size=[dict_size, word_dim],
            dtype='float32',
            is_sparse=IS_SPARSE)

        # use rnn unit to update rnn
        # TODO share parameter with trainer
        # current_state = pd.fc(input=[pre_ids_emb, pre_state_expanded],
        current_state = pd.fc(input=[pre_ids_emb],
                              size=decoder_size,
                              act='tanh')
        current_score = pd.fc(input=current_state,
                              size=target_dict_dim,
                              act='softmax')

        topk_scores, topk_indices = pd.topk(current_score, k=50)
        selected_ids, selected_scores = pd.beam_search(
            pre_ids, topk_indices, topk_scores, beam_size, end_id=10, level=0)

        pd.increment(x=counter, value=1, in_place=True)

        # update the memories
        pd.array_write(current_state, array=state_array, i=counter)
        pd.array_write(selected_ids, array=ids_array, i=counter)
        pd.array_write(selected_scores, array=scores_array, i=counter)

        pd.less_than(x=counter, y=array_len, cond=cond)

    translation_ids, translation_scores = pd.beam_search_decode(
        ids=ids_array, scores=scores_array)

    # return init_ids, init_scores

    return translation_ids, translation_scores


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
    rnn_out = decoder_train(context)
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
            if batch_id > 3:
                break
            batch_id += 1


def decode_main():
    # a newe block
    # with pd.BlockGuard() as block:
    context = encoder()
    # translation_ids, translation_scores = decoder_decode(context)
    translation_ids, translation_scores = decoder_decode(context)
    exe = Executor(place)
    exe.run(framework.default_startup_program())

    init_ids_data = np.array([1 for i in range(batch_size)], dtype='int64')
    init_scores_data = np.array(
        [1. for i in range(batch_size)], dtype='float32')
    init_ids_data = init_ids_data.reshape((batch_size, 1))
    init_scores_data = init_scores_data.reshape((batch_size, 1))
    init_lod = [i for i in range(batch_size)] + [batch_size]
    init_lod = [init_lod, init_lod]

    print 'init_ids', init_ids_data
    print 'init_scores', init_scores_data

    train_data = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.train(dict_size), buf_size=1000),
        batch_size=batch_size)
    for no, data in enumerate(train_data()):
        init_ids = set_init_lod(init_ids_data, init_lod, place)
        init_scores = set_init_lod(init_scores_data, init_lod, place)

        print 'init_ids.dims', init_ids.dtype()
        print 'init_scores.dims', init_scores.dtype()
        print np.array(init_ids)
        print np.array(init_scores)

        word_data = to_lodtensor(map(lambda x: x[0], data), place)
        # trg_word = to_lodtensor(map(lambda x: x[1], data), place)
        # trg_word_next = to_lodtensor(map(lambda x: x[2], data), place)
        exe.run(
            framework.default_main_program(),
            feed={
                'src_word_id': word_data,
                'init_ids': init_ids,
                'init_scores': init_scores
            },
            #fetch_list=['init_ids', 'init_scores']
        )
        #fetch_list=[translation_ids, translation_scores])
        break


if __name__ == '__main__':
    # train_main()
    decode_main()
