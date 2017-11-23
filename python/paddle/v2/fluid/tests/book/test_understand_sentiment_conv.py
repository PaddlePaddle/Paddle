import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid.core as core
import paddle.v2.fluid.evaluator as evaluator
import paddle.v2.fluid.framework as framework
import paddle.v2.fluid.layers as layers
import paddle.v2.fluid.nets as nets
from paddle.v2.fluid.executor import Executor
from paddle.v2.fluid.optimizer import AdamOptimizer


def convolution_net(input_dim, class_dim=2, emb_dim=32, hid_dim=32):
    data = layers.data(name="words", shape=[1], data_type="int64")
    label = layers.data(name="label", shape=[1], data_type="int64")

    emb = layers.embedding(input=data, size=[input_dim, emb_dim])
    conv_3 = nets.sequence_conv_pool(
        input=emb,
        num_filters=hid_dim,
        filter_size=3,
        act="tanh",
        pool_type="sqrt")
    conv_4 = nets.sequence_conv_pool(
        input=emb,
        num_filters=hid_dim,
        filter_size=4,
        act="tanh",
        pool_type="sqrt")
    prediction = layers.fc(input=[conv_3, conv_4],
                           size=class_dim,
                           act="softmax")
    cost = layers.cross_entropy(input=prediction, label=label)
    avg_cost = layers.mean(x=cost)
    adam_optimizer = AdamOptimizer(learning_rate=0.002)
    opts = adam_optimizer.minimize(avg_cost)
    accuracy, acc_out = evaluator.accuracy(input=prediction, label=label)
    return avg_cost, accuracy, acc_out


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

    cost, accuracy, acc_out = convolution_net(
        input_dim=dict_dim, class_dim=class_dim)

    train_data = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.imdb.train(word_dict), buf_size=1000),
        batch_size=BATCH_SIZE)
    place = core.CPUPlace()
    exe = Executor(place)

    exe.run(framework.default_startup_program())

    for pass_id in xrange(PASS_NUM):
        accuracy.reset(exe)
        for data in train_data():
            tensor_words = to_lodtensor(map(lambda x: x[0], data), place)

            label = np.array(map(lambda x: x[1], data)).astype("int64")
            label = label.reshape([BATCH_SIZE, 1])

            tensor_label = core.LoDTensor()
            tensor_label.set(label, place)

            outs = exe.run(framework.default_main_program(),
                           feed={"words": tensor_words,
                                 "label": tensor_label},
                           fetch_list=[cost, acc_out])
            cost_val = np.array(outs[0])
            acc_val = np.array(outs[1])
            pass_acc = accuracy.eval(exe)
            print("cost=" + str(cost_val) + " acc=" + str(acc_val) +
                  " pass_acc=" + str(pass_acc))
            if cost_val < 1.0 and pass_acc > 0.8:
                exit(0)
    exit(1)


if __name__ == '__main__':
    main()
