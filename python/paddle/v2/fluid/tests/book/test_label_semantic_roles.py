import math

import numpy as np
import paddle.v2 as paddle
import paddle.v2.dataset.conll05 as conll05
import paddle.v2.fluid.core as core
import paddle.v2.fluid.framework as framework
import paddle.v2.fluid.layers as layers
from paddle.v2.fluid.executor import Executor
from paddle.v2.fluid.optimizer import MomentumOptimizer

word_dict, verb_dict, label_dict = conll05.get_dict()
word_dict_len = len(word_dict)
label_dict_len = len(label_dict)
pred_len = len(verb_dict)

mark_dict_len = 2
word_dim = 32
mark_dim = 5
hidden_dim = 512
depth = 8
default_std = 1 / math.sqrt(hidden_dim) / 3.0
mix_hidden_lr = 1e-3

IS_SPARSE = True
PASS_NUM = 1
BATCH_SIZE = 20


def db_lstm():
    # 8 features
    word = layers.data(name='word_data', shape=[1], data_type='int64')
    predicate = layers.data(name='verb_data', shape=[1], data_type='int64')
    ctx_n2 = layers.data(name='ctx_n2_data', shape=[1], data_type='int64')
    ctx_n1 = layers.data(name='ctx_n1_data', shape=[1], data_type='int64')
    ctx_0 = layers.data(name='ctx_0_data', shape=[1], data_type='int64')
    ctx_p1 = layers.data(name='ctx_p1_data', shape=[1], data_type='int64')
    ctx_p2 = layers.data(name='ctx_p2_data', shape=[1], data_type='int64')
    mark = layers.data(name='mark_data', shape=[1], data_type='int64')

    emb_para = {'name': 'emb', 'initial_std': 0., 'is_static': True}
    std_0 = {'initial_std': 0.}
    std_default = {'initial_std': default_std}

    predicate_embedding = layers.embedding(
        input=predicate,
        size=[pred_len, word_dim],
        data_type='float32',
        is_sparse=IS_SPARSE,
        param_attr={'name': 'vemb',
                    'initial_std': default_std})

    mark_embedding = layers.embedding(
        input=mark,
        size=[mark_dict_len, mark_dim],
        data_type='float32',
        is_sparse=IS_SPARSE,
        param_attr=std_0)

    word_input = [word, ctx_n2, ctx_n1, ctx_0, ctx_p1, ctx_p2]
    emb_layers = [
        layers.embedding(
            size=[word_dict_len, word_dim], input=x, param_attr=emb_para)
        for x in word_input
    ]
    emb_layers.append(predicate_embedding)
    emb_layers.append(mark_embedding)

    hidden_0_layers = [
        layers.fc(input=emb,
                  size=hidden_dim,
                  bias_attr=std_default,
                  param_attr=std_default) for emb in emb_layers
    ]

    hidden_0 = layers.sums(input=hidden_0_layers)

    lstm_para_attr = {'initial_std': 0.0, 'learning_rate': 1.0}
    hidden_para_attr = {
        'initial_std': default_std,
        'learning_rate': mix_hidden_lr
    }

    lstm_0 = layers.dynamic_lstm(
        input=hidden_0,
        size=hidden_dim,
        candidate_activation='relu',
        gate_activation='sigmoid',
        cell_activation='sigmoid',
        bias_attr=std_0,
        param_attr=lstm_para_attr)

    # stack L-LSTM and R-LSTM with direct edges
    input_tmp = [hidden_0, lstm_0]

    for i in range(1, depth):
        mix_hidden = layers.sums(input=[
            layers.fc(input=input_tmp[0],
                      param_attr=hidden_para_attr,
                      size=hidden_dim,
                      bias_attr=std_default),
            layers.fc(input=input_tmp[1],
                      param_attr=lstm_para_attr,
                      size=hidden_dim,
                      bias_attr=std_default)
        ])

        lstm = layers.dynamic_lstm(
            input=mix_hidden,
            size=hidden_dim,
            candidate_activation='relu',
            gate_activation='sigmoid',
            cell_activation='sigmoid',
            is_reverse=((i % 2) == 1),
            bias_attr=std_0,
            param_attr=lstm_para_attr)

        input_tmp = [mix_hidden, lstm]

        feature_out = layers.sums(input=[
            layers.fc(input=input_tmp[0],
                      size=label_dict_len,
                      bias_attr=std_default,
                      param_attr=hidden_para_attr),
            layers.fc(input=input_tmp[1],
                      size=label_dict_len,
                      bias_attr=std_default,
                      param_attr=lstm_para_attr)
        ])

        return feature_out


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


def main():
    # define network topology
    feature_out = db_lstm()
    target = layers.data(name='target', shape=[1], data_type='int64')
    crf_cost = layers.linear_chain_crf(
        input=feature_out,
        label=target,
        param_attr={
            "name": 'crfw',
            "initial_std": default_std,
            "learning_rate": mix_hidden_lr
        })
    avg_cost = layers.mean(x=crf_cost)
    adam_optimizer = MomentumOptimizer(learning_rate=0.001, momentum=0.9)
    opts = adam_optimizer.minimize(avg_cost)

    train_data = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.conll05.test(), buf_size=8192),
        batch_size=BATCH_SIZE)
    place = core.CPUPlace()
    exe = Executor(place)

    exe.run(framework.default_startup_program())

    for pass_id in xrange(PASS_NUM):
        for data in train_data():
            word_data = to_lodtensor(map(lambda x: x[0], data), place)
            ctx_n2_data = to_lodtensor(map(lambda x: x[1], data), place)
            ctx_n1_data = to_lodtensor(map(lambda x: x[2], data), place)
            ctx_0_data = to_lodtensor(map(lambda x: x[3], data), place)
            ctx_p1_data = to_lodtensor(map(lambda x: x[4], data), place)
            ctx_p2_data = to_lodtensor(map(lambda x: x[5], data), place)
            verb_data = to_lodtensor(map(lambda x: x[6], data), place)
            mark_data = to_lodtensor(map(lambda x: x[7], data), place)
            target = to_lodtensor(map(lambda x: x[8], data), place)

            outs = exe.run(framework.default_main_program(),
                           feed={
                               'word_data': word_data,
                               'ctx_n2_data': ctx_n2_data,
                               'ctx_n1_data': ctx_n1_data,
                               'ctx_0_data': ctx_0_data,
                               'ctx_p1_data': ctx_p1_data,
                               'ctx_p2_data': ctx_p2_data,
                               'verb_data': verb_data,
                               'mark_data': mark_data,
                               'target': target
                           },
                           fetch_list=[avg_cost])
            avg_cost_val = np.array(outs[0])

            print("cost=" + str(avg_cost_val))


if __name__ == '__main__':
    main()
