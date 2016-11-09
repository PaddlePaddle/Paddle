from paddle.trainer_config_helpers import *

settings(batch_size=1000, learning_rate=1e-4)

din = data_layer(name='data', size=120)

outputs(
    grumemory(
        input=din,
        size=40,
        reverse=True,
        gate_act=TanhActivation(),
        act=SigmoidActivation()))
