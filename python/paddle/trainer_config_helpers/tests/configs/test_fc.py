from paddle.trainer_config_helpers import *

settings(batch_size=1000, learning_rate=1e-5)

din = data_layer(name='data', size=100)

trans = trans_layer(input=din)

hidden = fc_layer(input=trans, size=100, bias_attr=False)

mask = data_layer(name='mask', size=100)

hidden_sel = selective_fc_layer(
    input=din, select=mask, size=100, act=SigmoidActivation())

outputs(hidden, hidden_sel)
