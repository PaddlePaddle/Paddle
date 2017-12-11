import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import paddle.v2.fluid.core as core
import paddle.v2.fluid.framework as framework
import paddle.v2.fluid.layers as pd
from paddle.v2.fluid.executor import Executor

dict_size = 30000
source_dict_dim = target_dict_dim = dict_size
src_dict, trg_dict = paddle.dataset.wmt14.get_dict(dict_size)
hidden_dim = 32
word_dim = 16
IS_SPARSE = True
batch_size = 10
max_length = 8
topk_size = 50
trg_dic_size = 10000
beam_size = 10

decoder_size = hidden_dim


def encoder():
    # encoder
    src_word_id = pd.data(
        name="src_word_id", shape=[1], dtype='int64', lod_level=1)
    src_embedding = pd.embedding(
        input=src_word_id,
        size=[dict_size, word_dim],
        dtype='float32',
        is_sparse=IS_SPARSE,
        param_attr=fluid.ParamAttr(name='vemb'))

    fc1 = pd.fc(input=src_embedding, size=hidden_dim * 4, act='tanh')
    lstm_hidden0, lstm_0 = pd.dynamic_lstm(input=fc1, size=hidden_dim * 4)
    encoder_out = pd.sequence_pool(input=lstm_hidden0, pool_type="last")
    return encoder_out


def decoder_train(context):
    # decoder
    trg_language_word = pd.data(
        name="target_language_word", shape=[1], dtype='int64', lod_level=1)
    trg_embedding = pd.embedding(
        input=trg_language_word,
        size=[dict_size, word_dim],
        dtype='float32',
        is_sparse=IS_SPARSE,
        param_attr=fluid.ParamAttr(name='vemb'))

    rnn = pd.DynamicRNN()
    with rnn.block():
        current_word = rnn.step_input(trg_embedding)
        mem = rnn.memory(init=context)
        fc1 = pd.fc(input=[current_word, mem], size=decoder_size, act='tanh')
        out = pd.fc(input=fc1, size=target_dict_dim, act='softmax')
        rnn.update_memory(mem, fc1)
        rnn.output(out)

    return rnn()


def decoder_decode(context):
    init_state = context
    array_len = pd.fill_constant(shape=[1], dtype='int64', value=max_length)
    counter = pd.zeros(shape=[1], dtype='int64')
    # fill the first element with init_state
    mem_array = pd.create_array('float32')
    pd.array_write(init_state, array=mem_array, i=counter)

    # ids, scores as memory
    ids_array = pd.create_array('int32')
    scores_array = pd.create_array('float32')
    init_ids = pd.ones(shape=[batch_size, 1], dtype='int32')
    init_scores = pd.ones(shape=[batch_size, 1], dtype='float32')
    # init ids to [1..]
    # init scores to [1.]
    pd.array_write(init_ids, array=ids_array, i=counter)
    pd.array_write(init_scores, array=scores_array, i=counter)

    cond = pd.less_than(x=counter, y=array_len)

    while_op = pd.While(cond=cond)
    with while_op.block():
        pre_ids = pd.array_read(array=ids_array, i=counter)
        pre_state = pd.array_read(array=mem_array, i=counter)
        id = pd.array_read(array=ids_array, i=counter)
        target_word = pd.embedding(
            input=id,
            size=[dict_size, word_dim],
            dtype='float32',
            is_sparse=IS_SPARSE, )

        pre_state_expanded = pd.seq_expand(pre_state, target_word)
        print 'pre_state', pre_state
        print 'target_word', target_word
        print 'pre_state_expanded', pre_state_expanded

        # use rnn unit to update rnn
        # TODO share parameter with trainer
        updated_hidden = pd.fc(input=[target_word, pre_state_expanded],
                               size=hidden_dim,
                               act='tanh')
        scores = pd.fc(input=updated_hidden,
                       size=target_dict_dim,
                       act='softmax')

        topk_scores, topk_indices = pd.topk(scores, k=50)
        selected_ids, selected_scores = pd.beam_search(
            pre_ids, topk_indices, topk_scores, beam_size, end_id=1)

        # update the memories
        pd.array_write(updated_hidden, array=mem_array, i=counter)
        pd.array_write(selected_ids, array=ids_array, i=counter)
        pd.array_write(selected_scores, array=scores_array, i=counter)

    translation_ids, translation_scores = pd.beam_search_decode(
        ids=ids_array, scores=scores_array)

    return translation_ids, translation_scores


# def decoder(context):
#     '''
#     decoder for generation, only used in the inference stage.
#     '''
#     init_state = pd.fc(context, size=hidden_dim)

#     # directly use while_loop
#     # TODO add candidate set check
#     array_len = pd.fill_constant(shape=[1], dtype='int64', value=max_length)
#     counter = pd.zeros(shape=[1], dtype='int64')

#     # TODO(ChunweiYan) check the init_state should pass the gradient back
#     mem_array = pd.array_write(init_state, i=counter)
#     # TODO(ChunweiYan) should init to <s>
#     ids_array = pd.create_array('int32')
#     # TODO(ChunweiYan) should init to 1s
#     scores_array = pd.create_array('float32')

#     # TODO(ChunweiYan) another stop condition, check candidate set empty should be added
#     cond = pd.less_than(x=counter, y=array_len)

#     init_ids = pd.ones(shape=[batch_size, 1], dtype='int32')
#     init_scores = pd.ones(shape=[batch_size, 1], dtype='float32')

#     pd.array_write(init_ids, array=ids_array, i=counter)
#     pd.array_write(init_scores, array=scores_array, i=counter)

#     while_op = pd.While(cond=cond)
#     with while_op.block():
#         pre_state = pd.array_read(array=mem_array, i=counter)
#         # get the words of the previous step
#         id = pd.array_read(array=ids_array, i=counter)
#         target_word = pd.embedding(
#             input=id,
#             size=[dict_size, word_dim],
#             dtype='float32',
#             is_sparse=IS_SPARSE)

#         encoder_ctx_expanded = pd.seq_expand(context, target_word)

#         # use a simple gru
#         # gru_input = pd.fc(
#         #     act='sigmoid',
#         #     input=[target_word, encoder_ctx_expanded],
#         #     # gru has 2 gates and 1 x
#         #     size=3 * hidden_dim)

#         # gru_input = pd.fc(
#         #     act='sigmoid',
#         #     input=[target_word],
#         #     # gru has 2 gates and 1 x
#         #     size=3 * hidden_dim)

#         # updated_hidden, _, _ = pd.gru_unit(
#         #     input=gru_input, hidden=pre_state, size=hidden_dim)

#         # simplified RNN
#         updated_hidden = pd.fc(input=[pre_state, target_word],
#                                size=hidden_dim,
#                                act='tanh')

#         scores = pd.fc(updated_hidden, size=trg_dic_size, act='softmax')
#         # use pre_mem to predict candidates
#         scores = pd.fc(pre_state, size=dict_size, act='softmax')
#         topk_scores, topk_indices = pd.topk(scores, k=50)
#         selected_ids, selected_scores = pd.beam_search(topk_indices,
#                                                        topk_scores)

#         pd.array_write(updated_hidden, array=mem_array, i=counter)
#         pd.array_write(selected_ids, array=ids_array, i=counter)
#         pd.array_write(selected_scores, array=scores_array, i=counter)

#     while_op()

#     translation_ids, translation_scores = pd.beam_search_decode(
#         ids=ids_array, scores=scores_array)

#     return translation_ids, translation_scores


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
    context = encoder()
    # rnn_out = decoder_train(context)
    # translation_ids, translation_scores = decoder_decode(context)
    label = pd.data(
        name="target_language_next_word", shape=[1], dtype='int64', lod_level=1)
    # cost = pd.cross_entropy(input=rnn_out, label=label)
    # avg_cost = pd.mean(x=cost)

    # optimizer = fluid.optimizer.Adagrad(learning_rate=1e-4)
    # optimizer.minimize(avg_cost)

    train_data = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.train(dict_size), buf_size=1000),
        batch_size=batch_size)

    place = core.CPUPlace()
    exe = Executor(place)

    exe.run(framework.default_startup_program())

    batch_id = 0
    for pass_id in xrange(1):
        for data in train_data():
            word_data = to_lodtensor(map(lambda x: x[0], data), place)
            trg_word = to_lodtensor(map(lambda x: x[1], data), place)
            trg_word_next = to_lodtensor(map(lambda x: x[2], data), place)
            outs = exe.run(
                framework.default_main_program(),
                feed={
                    'src_word_id': word_data,
                    # 'target_language_word': trg_word,
                    # 'target_language_next_word': trg_word_next
                }, )
            #fetch_list=[avg_cost])
            # avg_cost_val = np.array(outs[0])
            # print('pass_id=' + str(pass_id) + ' batch=' + str(batch_id) +
            #       " avg_cost=" + str(avg_cost_val))
            if batch_id > 3:
                break
            batch_id += 1

    print 'to decode'
    for data in train_data():
        word_data = to_lodtensor(map(lambda x: x[0], data), place)
        trg_word = to_lodtensor(map(lambda x: x[1], data), place)
        trg_word_next = to_lodtensor(map(lambda x: x[2], data), place)
        outs = exe.run(framework.default_main_program(),
                       feed={'src_word_id': word_data, })


if __name__ == '__main__':
    main()
