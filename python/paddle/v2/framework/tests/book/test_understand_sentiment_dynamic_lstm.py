import paddle.v2 as paddle
import paddle.v2.framework.layers as layers
import paddle.v2.framework.nets as nets
import paddle.v2.framework.core as core
import paddle.v2.framework.optimizer as optimizer

from paddle.v2.framework.framework import Program, g_main_program, g_startup_program
from paddle.v2.framework.executor import Executor

import numpy as np


def stacked_lstm_net(input_dim,
                     class_dim=2,
                     emb_dim=128,
                     hid_dim=512,
                     stacked_num=3):
    assert stacked_num % 2 == 1
    data = layers.data(name="words", shape=[1], data_type="int64")
    label = layers.data(name="label", shape=[1], data_type="int64")

    emb = layers.embedding(input=data, size=[input_dim, emb_dim])
    # add bias attr

    # TODO(qijun) linear act
    fc1 = layers.fc(input=emb, size=hid_dim)
    lstm1, cell1 = layers.dynamic_lstm(input=fc1, size=hid_dim)

    inputs = [fc1, lstm1]

    for i in range(2, stacked_num + 1):
        fc = layers.fc(input=inputs, size=hid_dim)
        lstm, cell = layers.dynamic_lstm(
            input=fc, size=hid_dim, is_reverse=(i % 2) == 0)
        inputs = [fc, lstm]

    fc_last = layers.sequence_pool(input=inputs[0], pool_type='max')
    lstm_last = layers.sequence_pool(input=inputs[1], pool_type='max')

    prediction = layers.fc(input=[fc_last, lstm_last],
                           size=class_dim,
                           act='softmax')
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


def main():
    BATCH_SIZE = 100
    PASS_NUM = 5

    word_dict = paddle.dataset.imdb.word_dict()
    print "load word dict successfully"
    dict_dim = len(word_dict)
    class_dim = 2

    cost, acc = stacked_lstm_net(input_dim=dict_dim, class_dim=class_dim)

    train_data = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.imdb.train(word_dict), buf_size=1000),
        batch_size=BATCH_SIZE)
    place = core.CPUPlace()
    exe = Executor(place)

    exe.run(g_startup_program)

    for pass_id in xrange(PASS_NUM):
        for data in train_data():
            tensor_words = to_lodtensor(map(lambda x: x[0], data), place)

            label = np.array(map(lambda x: x[1], data)).astype("int64")
            label = label.reshape([BATCH_SIZE, 1])

            tensor_label = core.LoDTensor()
            tensor_label.set(label, place)

            outs = exe.run(g_main_program,
                           feed={"words": tensor_words,
                                 "label": tensor_label},
                           fetch_list=[cost, acc])
            cost_val = np.array(outs[0])
            acc_val = np.array(outs[1])

            print("cost=" + str(cost_val) + " acc=" + str(acc_val))
            if cost_val < 1.0 and acc_val > 0.7:
                exit(0)
    exit(1)


if __name__ == '__main__':
    main()
