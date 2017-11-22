import numpy as np
import paddle.v2 as paddle
import paddle.v2.dataset.conll05 as conll05
import paddle.v2.fluid.core as core
import paddle.v2.fluid.framework as framework
import paddle.v2.fluid.layers as layers
from paddle.v2.fluid.executor import Executor, g_scope
from paddle.v2.fluid.optimizer import SGDOptimizer

dict_size = 30000
source_dict_dim = target_dict_dim = dict_size
src_dict, trg_dict = paddle.dataset.wmt14.get_dict(dict_size)
hidden_dim = 512
word_dim = 512

src_word_id = layers.data(name="src_word_id", shape=[1], data_type='int64')
src_embedding = layers.embedding(
    input=src_word_id,
    size=[dict_size, word_dim],
    data_type='float32',
    is_sparse=IS_SPARSE,
    param_attr={'name': 'vemb'})


def encoder():

    lstm_0 = layers.dynamic_lstm(
        input=src_embedding,
        size=hidden_dim,
        candidate_activation='sigmoid',
        cell_activation='sigmoid')

    lstm_1 = layers.dynamic_lstm(
        input=src_embedding,
        size=hidden_dim,
        candidate_activation='sigmoid',
        cell_activation='sigmoid',
        is_reverse=True)

    bidirect_lstm_out = layers.concat(input=[lstm_0, lstm_1])

    return bidirect_lstm_out


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
    encoder_out = encoder()
    train_data = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.train(), buf_size=1000))
    place = core.CPUPlace()
    exe = Executor(place)

    exe.run(framework.default_startup_program())

    for pass_id in xrange(10):
        for data in train_data():
            word_data = to_lodtensor(mp(lambda x: x[0], data), place)
            outs = exe.run(framework.default_main_program(),
                           feed={'word_data': word_data, },
                           fetch_list=[encoder_out])


if __main__ == '__main__':
    main()
