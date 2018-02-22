from paddle.trainer_config_helpers import *

settings(batch_size=1000, learning_rate=1e-5)

data = data_layer(name='data', size=2560)

row_conv = row_conv_layer(input=data, context_len=19, act=ReluActivation())

outputs(row_conv)
