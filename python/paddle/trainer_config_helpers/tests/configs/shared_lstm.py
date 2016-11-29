from paddle.trainer_config_helpers import *

settings(learning_rate=1e-4, batch_size=1000)

data_1 = data_layer(name='data_a', size=100)
data_2 = data_layer(name='data_b', size=100)

mixed_param = ParamAttr(name='mixed_param')

with mixed_layer(size=400, bias_attr=False) as m1:
    m1 += full_matrix_projection(input=data_1, param_attr=mixed_param)

with mixed_layer(size=400, bias_attr=False) as m2:
    m2 += full_matrix_projection(input=data_2, param_attr=mixed_param)

lstm_param = ParamAttr(name='lstm_param')
lstm_bias = ParamAttr(name='lstm_bias', initial_mean=0., initial_std=0.)

lstm1 = lstmemory_group(
    input=m1,
    param_attr=lstm_param,
    lstm_bias_attr=lstm_bias,
    mixed_bias_attr=False)
lstm2 = lstmemory_group(
    input=m2,
    param_attr=lstm_param,
    lstm_bias_attr=lstm_bias,
    mixed_bias_attr=False)

softmax_param = ParamAttr(name='softmax_param')

predict = fc_layer(
    input=[last_seq(input=lstm1), last_seq(input=lstm2)],
    size=10,
    param_attr=[softmax_param, softmax_param],
    bias_attr=False,
    act=SoftmaxActivation())
outputs(
    classification_cost(
        input=predict, label=data_layer(
            name='label', size=10)))
