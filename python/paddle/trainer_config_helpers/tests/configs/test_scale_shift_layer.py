from paddle.trainer_config_helpers import *

settings(batch_size=1000, learning_rate=1e-5)

data = data_layer(name='data', size=100)

scale = scale_shift_layer(input=data)

scale_shift = scale_shift_layer(input=data, bias_attr=False)

outputs(scale, scale_shift)
