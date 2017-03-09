import sys
import math
import numpy as np
import paddle.v2 as paddle
import paddle.v2.dataset.conll05 as conll05


def db_lstm():
    word_dict, verb_dict, label_dict = conll05.get_dict()
    word_dict_len = len(word_dict)
    label_dict_len = len(label_dict)
    pred_len = len(verb_dict)

    mark_dict_len = 2
    word_dim = 32
    mark_dim = 5
    hidden_dim = 512
    depth = 8

    #8 features
    def d_type(size):
        return paddle.data_type.integer_value_sequence(size)

    word = paddle.layer.data(name='word_data', type=d_type(word_dict_len))
    predicate = paddle.layer.data(name='verb_data', type=d_type(pred_len))

    ctx_n2 = paddle.layer.data(name='ctx_n2_data', type=d_type(word_dict_len))
    ctx_n1 = paddle.layer.data(name='ctx_n1_data', type=d_type(word_dict_len))
    ctx_0 = paddle.layer.data(name='ctx_0_data', type=d_type(word_dict_len))
    ctx_p1 = paddle.layer.data(name='ctx_p1_data', type=d_type(word_dict_len))
    ctx_p2 = paddle.layer.data(name='ctx_p2_data', type=d_type(word_dict_len))
    mark = paddle.layer.data(name='mark_data', type=d_type(mark_dict_len))

    target = paddle.layer.data(name='target', type=d_type(label_dict_len))

    default_std = 1 / math.sqrt(hidden_dim) / 3.0

    emb_para = paddle.attr.Param(name='emb', initial_std=0., learning_rate=0.)
    std_0 = paddle.attr.Param(initial_std=0.)
    std_default = paddle.attr.Param(initial_std=default_std)

    predicate_embedding = paddle.layer.embedding(
        size=word_dim,
        input=predicate,
        param_attr=paddle.attr.Param(
            name='vemb', initial_std=default_std))
    mark_embedding = paddle.layer.embedding(
        size=mark_dim, input=mark, param_attr=std_0)

    word_input = [word, ctx_n2, ctx_n1, ctx_0, ctx_p1, ctx_p2]
    emb_layers = [
        paddle.layer.embedding(
            size=word_dim, input=x, param_attr=emb_para) for x in word_input
    ]
    emb_layers.append(predicate_embedding)
    emb_layers.append(mark_embedding)

    hidden_0 = paddle.layer.mixed(
        size=hidden_dim,
        bias_attr=std_default,
        input=[
            paddle.layer.full_matrix_projection(
                input=emb, param_attr=std_default) for emb in emb_layers
        ])

    mix_hidden_lr = 1e-3
    lstm_para_attr = paddle.attr.Param(initial_std=0.0, learning_rate=1.0)
    hidden_para_attr = paddle.attr.Param(
        initial_std=default_std, learning_rate=mix_hidden_lr)

    lstm_0 = paddle.layer.lstmemory(
        input=hidden_0,
        act=paddle.activation.Relu(),
        gate_act=paddle.activation.Sigmoid(),
        state_act=paddle.activation.Sigmoid(),
        bias_attr=std_0,
        param_attr=lstm_para_attr)

    #stack L-LSTM and R-LSTM with direct edges
    input_tmp = [hidden_0, lstm_0]

    for i in range(1, depth):
        mix_hidden = paddle.layer.mixed(
            size=hidden_dim,
            bias_attr=std_default,
            input=[
                paddle.layer.full_matrix_projection(
                    input=input_tmp[0], param_attr=hidden_para_attr),
                paddle.layer.full_matrix_projection(
                    input=input_tmp[1], param_attr=lstm_para_attr)
            ])

        lstm = paddle.layer.lstmemory(
            input=mix_hidden,
            act=paddle.activation.Relu(),
            gate_act=paddle.activation.Sigmoid(),
            state_act=paddle.activation.Sigmoid(),
            reverse=((i % 2) == 1),
            bias_attr=std_0,
            param_attr=lstm_para_attr)

        input_tmp = [mix_hidden, lstm]

    feature_out = paddle.layer.mixed(
        size=label_dict_len,
        bias_attr=std_default,
        input=[
            paddle.layer.full_matrix_projection(
                input=input_tmp[0], param_attr=hidden_para_attr),
            paddle.layer.full_matrix_projection(
                input=input_tmp[1], param_attr=lstm_para_attr)
        ], )

    crf_cost = paddle.layer.crf(size=label_dict_len,
                                input=feature_out,
                                label=target,
                                param_attr=paddle.attr.Param(
                                    name='crfw',
                                    initial_std=default_std,
                                    learning_rate=mix_hidden_lr))

    crf_dec = paddle.layer.crf_decoding(
        name='crf_dec_l',
        size=label_dict_len,
        input=feature_out,
        label=target,
        param_attr=paddle.attr.Param(name='crfw'))

    return crf_cost, crf_dec


def load_parameter(file_name, h, w):
    with open(file_name, 'rb') as f:
        f.read(16)  # skip header.
        return np.fromfile(f, dtype=np.float32).reshape(h, w)


def main():
    paddle.init(use_gpu=False, trainer_count=1)

    # define network topology
    crf_cost, crf_dec = db_lstm()

    # create parameters
    parameters = paddle.parameters.create([crf_cost, crf_dec])

    # create optimizer
    optimizer = paddle.optimizer.Momentum(
        momentum=0,
        learning_rate=2e-2,
        regularization=paddle.optimizer.L2Regularization(rate=8e-4),
        model_average=paddle.optimizer.ModelAverage(
            average_window=0.5, max_average_window=10000), )

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "Pass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)

    trainer = paddle.trainer.SGD(cost=crf_cost,
                                 parameters=parameters,
                                 update_equation=optimizer)
    parameters.set('emb', load_parameter(conll05.get_embedding(), 44068, 32))

    trn_reader = paddle.batch(
        paddle.reader.shuffle(
            conll05.test(), buf_size=8192), batch_size=10)

    feeding = {
        'word_data': 0,
        'ctx_n2_data': 1,
        'ctx_n1_data': 2,
        'ctx_0_data': 3,
        'ctx_p1_data': 4,
        'ctx_p2_data': 5,
        'verb_data': 6,
        'mark_data': 7,
        'target': 8
    }

    trainer.train(
        reader=trn_reader,
        event_handler=event_handler,
        num_passes=10000,
        feeding=feeding)


if __name__ == '__main__':
    main()
