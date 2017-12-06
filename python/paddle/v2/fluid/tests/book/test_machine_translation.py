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
hidden_dim = 512
word_dim = 512
IS_SPARSE = True
batch_size = 50
max_length = 50
topk_size = 50
trg_dic_size = 10000

decoder_size = 512

src_word_id = layers.data(name="src_word_id", shape=[1], dtype='int64')
src_embedding = layers.embedding(
    input=src_word_id,
    size=[dict_size, word_dim],
    dtype='float32',
    is_sparse=IS_SPARSE,
    param_attr=fluid.ParamAttr(name='vemb'))


def encoder():

    lstm_hidden0, lstm_0 = layers.dynamic_lstm(
        input=src_embedding,
        size=hidden_dim,
        candidate_activation='sigmoid',
        cell_activation='sigmoid')

    lstm_hidden1, lstm_1 = layers.dynamic_lstm(
        input=src_embedding,
        size=hidden_dim,
        candidate_activation='sigmoid',
        cell_activation='sigmoid',
        is_reverse=True)

    bidirect_lstm_out = layers.concat([lstm_hidden0, lstm_hidden1], axis=0)

    return bidirect_lstm_out


def decoder_trainer(context):
    '''
    decoder with trainer
    '''
    trg_language_word = layers.data(
        name="target_language_word", shape=[1], dtype='int64')
    trg_embedding = layers.embedding(
        input=trg_language_word,
        size=[dict_size, word_dim],
        dtype='float32',
        is_sparse=IS_SPARSE,
        param_attr=fluid.ParamAttr(name='vemb'))

    encoded_proj = layers.fc(size=decoder_size, bias_attr=False, input=context)

    rnn = fluid.layers.DynamicRNN()

    with rnn.block():
        current_word = rnn.step_input(trg_embedding)
        encoded_proj_in = rnn.step_input(encoded_proj)
        mem = rnn.memory(shape=[decoder_size], dtype='float32')
        out_ = fluid.layers.fc(input=[encoded_proj_in, current_word, mem],
                               size=target_dict_dim,
                               act='softmax')
        rnn.update_memory(mem, out_)
        rnn.output(out_)

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
    encoder_out = encoder()
    decoder_out = decoder_trainer(encoder_out)
    label = layers.data(
        name="target_language_next_word", shape=[1], dtype='int64')

    cost = layers.cross_entropy(input=decoder_out, label=label)

    avg_cost = fluid.layers.mean(x=cost)

    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    optimizer.minimize(avg_cost)
    accuracy = fluid.evaluator.Accuracy(input=decoder_out, label=label)

    train_data = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.train(8000), buf_size=1000),
        batch_size=batch_size)

    place = core.CPUPlace()
    exe = Executor(place)

    exe.run(framework.default_startup_program())

    batch_id = 0
    for pass_id in xrange(2):
        print 'pass_id', pass_id
        for data in train_data():
            print 'batch', batch_id
            batch_id += 1
            if batch_id > 10: break
            word_data = to_lodtensor(map(lambda x: x[0], data), place)
            outs = exe.run(
                framework.default_main_program(),
                feed={'src_word_id': word_data, },
                fetch_list=[encoder_out, decoder_out, accuracy.metrics[0]])


if __name__ == '__main__':
    main()
