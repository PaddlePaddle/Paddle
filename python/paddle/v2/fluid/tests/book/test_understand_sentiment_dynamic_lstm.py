import math
import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid as fluid
from paddle.v2.fluid.param_attr import ParamAttr
from paddle.v2.fluid.initializer import NormalInitializer


def stacked_lstm_net(data,
                     label,
                     input_dim,
                     class_dim=2,
                     emb_dim=128,
                     hid_dim=512,
                     stacked_num=3,
                     batch_size=100):
    assert stacked_num % 2 == 1

    emb = fluid.layers.embedding(
        input=data,
        size=[input_dim, emb_dim],
        param_attr=ParamAttr(initializer=NormalInitializer(
            loc=0., scale=1.0 / math.sqrt(input_dim))))

    fc1 = fluid.layers.fc(input=emb,
                          size=hid_dim,
                          bias_attr=ParamAttr(initializer=NormalInitializer(
                              loc=0., scale=0.)),
                          param_attr=ParamAttr(
                              name='fc1',
                              initializer=NormalInitializer(
                                  loc=0., scale=1.0 / math.sqrt(emb_dim))))
    lstm1, cell1 = fluid.layers.dynamic_lstm(
        input=fc1,
        size=hid_dim,
        candidate_activation='relu',
        bias_attr=ParamAttr(initializer=NormalInitializer(
            loc=0., scale=0.)),
        param_attr=ParamAttr(initializer=NormalInitializer(
            loc=0., scale=1.0 / math.sqrt(emb_dim))))

    inputs = [fc1, lstm1]

    for i in range(2, stacked_num + 1):
        fc = fluid.layers.fc(input=inputs,
                             size=hid_dim,
                             bias_attr=ParamAttr(initializer=NormalInitializer(
                                 loc=0., scale=0.)),
                             param_attr=[
                                 ParamAttr(
                                     learning_rate=1e-3,
                                     initializer=NormalInitializer(
                                         loc=0., scale=1.0 /
                                         math.sqrt(hid_dim))), ParamAttr(
                                             learning_rate=1.,
                                             initializer=NormalInitializer(
                                                 loc=0., scale=0.))
                             ])
        lstm, cell = fluid.layers.dynamic_lstm(
            input=fc,
            size=hid_dim,
            is_reverse=(i % 2) == 0,
            candidate_activation='relu',
            bias_attr=ParamAttr(initializer=NormalInitializer(
                loc=0., scale=0.)),
            param_attr=ParamAttr(initializer=NormalInitializer(
                loc=0., scale=1.0 / math.sqrt(emb_dim))))
        inputs = [fc, lstm]

    fc_last = fluid.layers.sequence_pool(input=inputs[0], pool_type='max')
    lstm_last = fluid.layers.sequence_pool(input=inputs[1], pool_type='max')

    prediction = fluid.layers.fc(
        input=[fc_last, lstm_last],
        size=class_dim,
        bias_attr=ParamAttr(initializer=NormalInitializer(
            loc=0., scale=0.)),
        param_attr=[
            ParamAttr(
                learning_rate=1e-3,
                initializer=NormalInitializer(
                    loc=0., scale=1.0 / math.sqrt(hid_dim))), ParamAttr(
                        learning_rate=1.,
                        initializer=NormalInitializer(
                            loc=0., scale=0.))
        ],
        act='softmax')
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    avg_cost = fluid.layers.scale(x=avg_cost, scale=float(batch_size))
    adam_optimizer = fluid.optimizer.Adam(learning_rate=0.002)
    adam_optimizer.minimize(avg_cost)
    accuracy = fluid.evaluator.Accuracy(input=prediction, label=label)
    return avg_cost, accuracy, accuracy.metrics[0]


def to_lodtensor(data, place):
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
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

    data = fluid.layers.data(
        name="words", shape=[1], dtype="int64", lod_level=1)
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
    cost, accuracy, acc_out = stacked_lstm_net(
        data,
        label,
        input_dim=dict_dim,
        class_dim=class_dim,
        batch_size=BATCH_SIZE)

    train_data = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.imdb.train(word_dict), buf_size=1000),
        batch_size=BATCH_SIZE)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(feed_list=[data, label], place=place)

    exe.run(fluid.default_startup_program())

    for pass_id in xrange(PASS_NUM):
        accuracy.reset(exe)
        for data in train_data():
            cost_val, acc_val = exe.run(fluid.default_main_program(),
                                        feed=feeder.feed(data),
                                        fetch_list=[cost, acc_out])
            pass_acc = accuracy.eval(exe)
            print("cost=" + str(cost_val) + " acc=" + str(acc_val) +
                  " pass_acc=" + str(pass_acc))
            if cost_val < 1.0 and acc_val > 0.8:
                exit(0)
    exit(1)


if __name__ == '__main__':
    main()
