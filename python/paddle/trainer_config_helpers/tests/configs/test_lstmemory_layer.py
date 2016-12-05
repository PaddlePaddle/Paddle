from paddle.trainer_config_helpers import *

settings(batch_size=1000, learning_rate=1e-5)

din = data_layer(name='data', size=128)

outputs(
    lstmemory(
        input=din,
        reverse=True,
        gate_act=TanhActivation(),
        act=TanhActivation(),
        size=32))
