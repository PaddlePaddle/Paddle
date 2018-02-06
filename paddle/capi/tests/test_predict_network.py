from paddle.trainer_config_helpers import *

settings(batch_size=100)

x = data_layer(name='x', size=100)

y = fc_layer(
    input=x,
    size=100,
    bias_attr=ParamAttr(name='b'),
    param_attr=ParamAttr(name='w'))

outputs(y)
