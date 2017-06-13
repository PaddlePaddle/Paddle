from paddle.trainer_config_helpers import *

settings(batch_size=1000, learning_rate=1e-5)

index = data_layer(name='index', size=1)
din1 = data_layer(name='data1', size=30)
din2 = data_layer(name='data2', size=30)
din3 = data_layer(name='data3', size=30)

dout = multiplex_layer([index, din1, din2, din3])

outputs(dout)
