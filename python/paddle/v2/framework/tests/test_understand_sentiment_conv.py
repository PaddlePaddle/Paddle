import paddle.v2 as paddle
import paddle.v2.framework.layers as layers
import paddle.v2.framework.nets as nets
import paddle.v2.framework.core as core
import paddle.v2.framework.optimizer as optimizer

from paddle.v2.framework.framework import Program, g_program, g_init_program
from paddle.v2.framework.executor import Executor

import numpy as np


def convolution_net(input_dim, class_dim=2, emb_dim=128, hid_dim=128):
    data = layers.data(name="words", shape=[1], data_type="int64")
    label = layers.data(name="label", shape=[1], data_type="int64")

    emb = layers.embedding(input=data, size=[input_dim, emb_dim])
    conv_3 = nets.sequence_conv_pool(
        input=emb,
        num_filters=hid_dim,
        filter_size=3,
        pool_type="max",
        act="tanh")
    conv_4 = nets.sequence_conv_pool(
        input=emb,
        num_filters=hid_dim,
        filter_size=4,
        pool_type="max",
        act="tanh")
    prediction = layers.fc(input=[conv_3, conv_4],
                           size=class_dim,
                           act="softmax")

    cost = layers.cross_entropy(input=prediction, label=label)
    acc = layers.accuracy(input=prediction, label=label)
    return cost, acc


def to_lodtensor(data, place):
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0)
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = core.LoDTensor()
    res.set(flattened_data, place)
    #import pdb
    # pdb.set_trace()
    res.set_lod([lod])
    return res


def main():
    BATCH_SIZE = 100
    PASS_NUM = 3

    word_dict = paddle.dataset.imdb.word_dict()
    dict_dim = len(word_dict)
    class_dim = 2

    cost, acc = convolution_net(input_dim=dict_dim, class_dim=class_dim)

    train_data = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.imdb.train(word_dict), buf_size=1000),
        batch_size=BATCH_SIZE)
    place = core.CPUPlace()
    exe = Executor(place)

    exe.run(g_init_program)

    for pass_id in xrange(PASS_NUM):
        for data in train_data():
            tensor_words = to_lodtensor(map(lambda x: x[0], data), place)

            label = np.array(map(lambda x: x[1], data)).astype("int64")
            label = label.reshape([BATCH_SIZE, 1])

            tensor_label = core.LoDTensor()
            tensor_label.set(label, place)

            outs = exe.run(g_program,
                           feed={"words": tensor_words,
                                 "label": tensor_label},
                           fetch_list=[cost, acc])
            loss = np.array(outs[0])
            acc = np.array(outs[1])

            print("loss=" + str(loss) + " acc=" + str(acc))


if __name__ == '__main__':
    main()
