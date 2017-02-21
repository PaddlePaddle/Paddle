from paddle.trainer_config_helpers import *

settings(batch_size=1000, learning_rate=1e-5)

din1 = data_layer(name='data1', size=30)

din2 = data_layer(name='data2', size=30)

outputs(seq_concat_layer(a=din1, b=din2))
