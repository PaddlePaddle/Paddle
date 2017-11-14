import paddle.v2 as paddle
import paddle.v2.framework.layers as layers
import paddle.v2.framework.core as core
import paddle.v2.framework.optimizer as optimizer

from paddle.v2.framework.framework import g_main_program, g_startup_program
from paddle.v2.framework.executor import Executor

import numpy as np


def lstm_net(dict_dim, class_dim=2, emb_dim=32, seq_len=80, batch_size=50):
    data = layers.data(
        name="words",
        shape=[seq_len * batch_size, 1],
        append_batch_size=False,
        data_type="int64")
    label = layers.data(
        name="label",
        shape=[batch_size, 1],
        append_batch_size=False,
        data_type="int64")

    emb = layers.embedding(input=data, size=[dict_dim, emb_dim])
    emb = layers.reshape(x=emb, shape=[batch_size, seq_len, emb_dim])
    emb = layers.transpose(x=emb, axis=[1, 0, 2])

    c_pre_init = layers.fill_constant(
        dtype=emb.data_type, shape=[batch_size, emb_dim], value=0.0)
    layer_1_out = layers.lstm(emb, c_pre_init=c_pre_init, hidden_dim=emb_dim)
    layer_1_out = layers.transpose(x=layer_1_out, axis=[1, 0, 2])

    prediction = layers.fc(input=layer_1_out, size=class_dim, act="softmax")
    cost = layers.cross_entropy(input=prediction, label=label)

    avg_cost = layers.mean(x=cost)
    adam_optimizer = optimizer.AdamOptimizer(learning_rate=0.002)
    opts = adam_optimizer.minimize(avg_cost)
    acc = layers.accuracy(input=prediction, label=label)

    return avg_cost, acc


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


def chop_data(data, chop_len=80, batch_len=50):
    data = [(x[0][:chop_len], x[1]) for x in data if len(x[0]) >= chop_len]

    return data[:batch_len]


def prepare_feed_data(data, place):
    tensor_words = to_lodtensor(map(lambda x: x[0], data), place)

    label = np.array(map(lambda x: x[1], data)).astype("int64")
    label = label.reshape([50, 1])
    tensor_label = core.LoDTensor()
    tensor_label.set(label, place)

    return tensor_words, tensor_label


def main():
    word_dict = paddle.dataset.imdb.word_dict()
    cost, acc = lstm_net(dict_dim=len(word_dict), class_dim=2)

    batch_size = 100
    train_data = paddle.batch(
        paddle.reader.buffered(
            paddle.dataset.imdb.train(word_dict), size=batch_size * 10),
        batch_size=batch_size)

    data = chop_data(next(train_data()))

    place = core.CPUPlace()
    tensor_words, tensor_label = prepare_feed_data(data, place)
    exe = Executor(place)
    exe.run(g_startup_program)

    while True:
        outs = exe.run(g_main_program,
                       feed={"words": tensor_words,
                             "label": tensor_label},
                       fetch_list=[cost, acc])
        cost_val = np.array(outs[0])
        acc_val = np.array(outs[1])

        print("cost=" + str(cost_val) + " acc=" + str(acc_val))
        if acc_val > 0.9:
            break


if __name__ == '__main__':
    main()
