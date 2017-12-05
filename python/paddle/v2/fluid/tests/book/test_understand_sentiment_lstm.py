import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid as fluid


def lstm_net(dict_dim, class_dim=2, emb_dim=32, seq_len=80, batch_size=50):
    data = fluid.layers.data(
        name="words",
        shape=[seq_len * batch_size, 1],
        append_batch_size=False,
        dtype="int64",
        lod_level=1)
    label = fluid.layers.data(
        name="label",
        shape=[batch_size, 1],
        append_batch_size=False,
        dtype="int64")

    emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])
    emb = fluid.layers.reshape(x=emb, shape=[batch_size, seq_len, emb_dim])
    emb = fluid.layers.transpose(x=emb, axis=[1, 0, 2])

    c_pre_init = fluid.layers.fill_constant(
        dtype=emb.dtype, shape=[batch_size, emb_dim], value=0.0)
    c_pre_init.stop_gradient = False
    layer_1_out = fluid.layers.lstm(
        emb, c_pre_init=c_pre_init, hidden_dim=emb_dim)
    layer_1_out = fluid.layers.transpose(x=layer_1_out, axis=[1, 0, 2])

    prediction = fluid.layers.fc(input=layer_1_out,
                                 size=class_dim,
                                 act="softmax")
    cost = fluid.layers.cross_entropy(input=prediction, label=label)

    avg_cost = fluid.layers.mean(x=cost)
    adam_optimizer = fluid.optimizer.Adam(learning_rate=0.002)
    adam_optimizer.minimize(avg_cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)

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
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res


def chop_data(data, chop_len=80, batch_size=50):
    data = [(x[0][:chop_len], x[1]) for x in data if len(x[0]) >= chop_len]

    return data[:batch_size]


def prepare_feed_data(data, place):
    tensor_words = to_lodtensor(map(lambda x: x[0], data), place)

    label = np.array(map(lambda x: x[1], data)).astype("int64")
    label = label.reshape([len(label), 1])
    tensor_label = fluid.LoDTensor()
    tensor_label.set(label, place)

    return tensor_words, tensor_label


def main():
    BATCH_SIZE = 100
    PASS_NUM = 5

    word_dict = paddle.dataset.imdb.word_dict()
    print "load word dict successfully"
    dict_dim = len(word_dict)
    class_dim = 2

    cost, acc = lstm_net(dict_dim=dict_dim, class_dim=class_dim)

    train_data = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.imdb.train(word_dict), buf_size=BATCH_SIZE * 10),
        batch_size=BATCH_SIZE)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    exe.run(fluid.default_startup_program())

    for pass_id in xrange(PASS_NUM):
        for data in train_data():
            chopped_data = chop_data(data)
            tensor_words, tensor_label = prepare_feed_data(chopped_data, place)

            outs = exe.run(fluid.default_main_program(),
                           feed={"words": tensor_words,
                                 "label": tensor_label},
                           fetch_list=[cost, acc])
            cost_val = np.array(outs[0])
            acc_val = np.array(outs[1])

            print("cost=" + str(cost_val) + " acc=" + str(acc_val))
            if acc_val > 0.7:
                exit(0)
    exit(1)


if __name__ == '__main__':
    main()
