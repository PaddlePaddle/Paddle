import numpy as np
import paddle.v2 as paddle
import paddle.v2.dataset.conll05 as conll05
import paddle.v2.fluid.core as core
import paddle.v2.fluid.framework as framework
import paddle.v2.fluid.layers as layers
from paddle.v2.fluid.executor import Executor, g_scope
from paddle.v2.fluid.optimizer import SGDOptimizer
import paddle.v2.fluid as fluid
import paddle.v2.fluid.layers as pd

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
    # TODO(ChunweiYan) add implementation after dynamic rnn is ready
    pass


def decoder(context):
    '''
    decoder for generation, only used in the inference stage.
    '''
    init_state = fluid.layers.fc(context, size=hidden_dim)

    # directly use while_loop
    # TODO add candidate set check
    array_len = layers.fill_constant(shape=[1], dtype='int64', value=max_length)
    counter = layers.zeros(shape=[1], dtype='int64')

    # TODO(ChunweiYan) check the init_state should pass the gradient back
    mem_array = pd.array_write(init_state, i=counter)
    # TODO(ChunweiYan) should init to <s>
    ids_array = pd.create_array('int32')
    # TODO(ChunweiYan) should init to 1s
    scores_array = pd.create_array('float32')

    # TODO(ChunweiYan) another stop condition, check candidate set empty should be added
    cond = layers.less_than(x=counter, y=array_len)

    while_op = layers.While(cond=cond)
    with while_op.block():
        pre_state = pd.array_read(array=mem_array, i=counter)
        # get the words of the previous step
        target_word = pd.embedding(
            input=ids_array,
            size=[dict_size, word_dim],
            dtype='float32',
            is_sparse=IS_SPARSE)

        encoder_ctx_expanded = pd.seq_expand(context, target_word)

        # use a simple gru
        gru_input = pd.fc(
            act='sigmoid',
            input=pd.sum([target_word, encoder_ctx_expanded]),
            # gru has 2 gates and 1 x
            size=3 * hidden_dim)
        updated_hidden, _, _ = gru_unit(
            input=gru_input, hidden=pre_state, size=hidden_dim)
        scores = pd.fc(updated_hidden,
                       size=trg_dic_size,
                       bias=None,
                       activation='softmax')
        # use pre_mem to predict candidates
        scores = pd.fc(pre_state, size=dict_size, act='softmax')
        topk_scores, topk_indices = pd.topk(scores, k=50)
        selected_ids, selected_scores = pd.beam_search(topk_indices,
                                                       topk_scores)

        # update the memories
        pd.array_write(updated_hidden, array=mem_array, i=counter)
        pd.array_write(selected_ids, array=ids_array, i=counter)
        pd.array_write(selected_scores, array=scores_array, i=counter)

    while_op()

    translation_ids, translation_scores = pd.beam_search_decode(
        ids=ids_array, scores=scores_array)

    return translation_ids, translation_scores


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
    ids, scores = decoder(encoder_out)

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
            outs = exe.run(framework.default_main_program(),
                           feed={'src_word_id': word_data, },
                           fetch_list=[encoder_out])


if __name__ == '__main__':
    main()
