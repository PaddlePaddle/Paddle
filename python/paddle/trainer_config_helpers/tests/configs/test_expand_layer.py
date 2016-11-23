from paddle.trainer_config_helpers import *

settings(batch_size=1000, learning_rate=1e-5)

din = data_layer(name='data', size=30)
data_seq = data_layer(name='data_seq', size=30)

outputs(
    expand_layer(
        input=din, expand_as=data_seq, expand_level=ExpandLevel.FROM_SEQUENCE),
    expand_layer(
        input=din, expand_as=data_seq, expand_level=ExpandLevel.FROM_TIMESTEP))
