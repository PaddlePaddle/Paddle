from paddle.trainer_config_helpers import *

data = data_layer(name='data', size=100)

scale = scale_shift_layer(input=data, bias_attr=False)

scale_shift = scale_shift_layer(input=data)

outputs(scale, scale_shift)
