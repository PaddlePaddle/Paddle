import paddle.v2 as paddle
import paddle.v2.framework.layers as layers
import paddle.v2.framework.nets as nets
import paddle.v2.framework.core as core
import paddle.v2.framework.optimizer as optimizer

from paddle.v2.framework.framework import Program, g_program, g_init_program
from paddle.v2.framework.executor import Executor

import pdb
import numpy as np

# FIXME: parameter changed.


def convolution_net(input_dim, class_dim=2, emb_dim=32, hid_dim=16):
    data = layers.data(name="words", shape=[1], data_type="int64")
    label = layers.data(name="label", shape=[1], data_type="int64")

    emb = layers.embedding(input=data, size=[input_dim, emb_dim])
    conv_3 = nets.sequence_conv_pool(
        input=emb,
        num_filters=hid_dim,
        filter_size=3,
        act="tanh",
        pool_type="MAX")
    conv_5 = nets.sequence_conv_pool(
        input=emb,
        num_filters=hid_dim,
        filter_size=5,
        act="tanh",
        pool_type="MAX")
    prediction = layers.fc(input=[conv_3, conv_5],
                           size=class_dim,
                           act="softmax")
    #prediction = layers.softmax(x=before_prediction)
    cost = layers.cross_entropy(input=prediction, label=label)
    avg_cost = layers.mean(x=cost)
    adam_optimizer = optimizer.AdamOptimizer(learning_rate=0.002)
    opts = adam_optimizer.minimize(avg_cost)
    acc = layers.accuracy(input=prediction, label=label)
    return avg_cost, acc, prediction, conv_3, conv_5, emb


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
    dict_dim = len(word_dict)
    class_dim = 2

    cost, acc, prediction, conv_3, conv_5, emb = convolution_net(
        input_dim=dict_dim, class_dim=class_dim)

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

            #var_name = []
            #var_vars = []
            # for k, v in g_program.global_block().vars.items():
            #    if v.type == core.VarDesc.VarType.LOD_TENSOR:
            #        var_name.append(k)
            #        var_vars.append(v)
            var_names = [
                "embedding_0.w_0", "embedding_0.w_0@GRAD",
                "sequence_conv_0.w_0", "embedding_0.tmp_0",
                "sequence_conv_0.tmp_0", "sequence_conv_0.w_0@GRAD",
                "embedding_0.tmp_0@GRAD", "sequence_conv_0.tmp_0@GRAD",
                "sequence_conv_0.tmp_1", "sequence_conv_0.tmp_1@GRAD",
                "sequence_pool_0.tmp_0", "sequence_pool_0.tmp_0@GRAD"
            ]
            var_vars = [cost]
            for n in var_names:
                var_vars.append(g_program.global_block().var(n))

            var_names = ["mean_0.tmp_0"] + var_names

            outs = exe.run(g_program,
                           feed={"words": tensor_words,
                                 "label": tensor_label},
                           fetch_list=var_vars)
            out_vals = map(lambda x: np.array(x), outs)

            ccc = dict(zip(var_names, out_vals))

            print("cost=" + str(ccc["mean_0.tmp_0"]))
            #print("pre=" + str(pre_val))

            pdb.set_trace()


if __name__ == '__main__':
    main()
