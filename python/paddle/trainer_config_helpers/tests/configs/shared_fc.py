from paddle.trainer_config_helpers import *

settings(learning_rate=1e-4, batch_size=1000)

a = data_layer(name='feature_a', size=200)
b = data_layer(name='feature_b', size=200)

fc_param = ParamAttr(name='fc_param', initial_max=1.0, initial_min=-1.0)
bias_param = ParamAttr(name='bias_param', initial_mean=0.0, initial_std=0.0)

softmax_param = ParamAttr(
    name='softmax_param', initial_max=1.0, initial_min=-1.0)

hidden_a = fc_layer(
    input=a, size=200, param_attr=fc_param, bias_attr=bias_param)
hidden_b = fc_layer(
    input=b, size=200, param_attr=fc_param, bias_attr=bias_param)

predict = fc_layer(
    input=[hidden_a, hidden_b],
    param_attr=[softmax_param, softmax_param],
    bias_attr=False,
    size=10,
    act=SoftmaxActivation())

outputs(
    classification_cost(
        input=predict, label=data_layer(
            name='label', size=10)))
