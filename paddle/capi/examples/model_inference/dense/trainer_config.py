from paddle.trainer_config_helpers import *

img = data_layer(name='pixel', size=784)

hidden = fc_layer(
    input=img,
    size=200,
    param_attr=ParamAttr(name='hidden.w'),
    bias_attr=ParamAttr(name='hidden.b'))

prob = fc_layer(
    input=hidden,
    size=10,
    act=SoftmaxActivation(),
    param_attr=ParamAttr(name='prob.w'),
    bias_attr=ParamAttr(name='prob.b'))

outputs(prob)
