import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import paddle.v2.fluid.core as core
import paddle.v2.fluid.framework as framework
import paddle.v2.fluid.layers as layers
from paddle.v2.fluid.executor import Executor

dict_size = 30000
source_dict_dim = target_dict_dim = dict_size
src_dict, trg_dict = paddle.dataset.wmt14.get_dict(dict_size)
hidden_dim = 32
word_dim = 16
IS_SPARSE = True
batch_size = 10
max_length = 50
topk_size = 50
trg_dic_size = 10000

decoder_size = hidden_dim


def encoder_decoder():
    # encoder
    src_word_id = layers.data(
        name="src_word_id", shape=[1], dtype='int64', lod_level=1)
    src_embedding = layers.embedding(
        input=src_word_id,
        size=[dict_size, word_dim],
        dtype='float32',
        is_sparse=IS_SPARSE,
        param_attr=fluid.ParamAttr(name='vemb'))

    fc1 = fluid.layers.fc(input=src_embedding, size=hidden_dim * 4, act='tanh')
    lstm_hidden0, lstm_0 = layers.dynamic_lstm(input=fc1, size=hidden_dim * 4)
    encoder_out = layers.sequence_last_step(input=lstm_hidden0)

    # decoder
    trg_language_word = layers.data(
        name="target_language_word", shape=[1], dtype='int64', lod_level=1)
    trg_embedding = layers.embedding(
        input=trg_language_word,
        size=[dict_size, word_dim],
        dtype='float32',
        is_sparse=IS_SPARSE,
        param_attr=fluid.ParamAttr(name='vemb'))

    rnn = fluid.layers.DynamicRNN()
    with rnn.block():
        current_word = rnn.step_input(trg_embedding)
        mem = rnn.memory(init=encoder_out)
        fc1 = fluid.layers.fc(input=[current_word, mem],
                              size=decoder_size,
                              act='tanh')
        out = fluid.layers.fc(input=fc1, size=target_dict_dim, act='softmax')
        rnn.update_memory(mem, fc1)
        rnn.output(out)

    return rnn()


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
    rnn_out = encoder_decoder()
    label = layers.data(
        name="target_language_next_word", shape=[1], dtype='int64', lod_level=1)
    cost = layers.cross_entropy(input=rnn_out, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    optimizer = fluid.optimizer.Adagrad(learning_rate=1e-4)
    optimizer.minimize(avg_cost)

    train_data = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.train(dict_size), buf_size=1000),
        batch_size=batch_size)

    place = core.CPUPlace()
    exe = Executor(place)

    exe.run(framework.default_startup_program())

    batch_id = 0
    for pass_id in xrange(2):
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
                exit(0)
            batch_id += 1


if __name__ == '__main__':
    main()
