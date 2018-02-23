from paddle.trainer_config_helpers import *

settings(batch_size=1000, learning_rate=1e-5)

din = data_layer(name='data', size=30)

outputs(
    repeat_layer(
        input=din, num_repeats=10, as_row_vector=True),
    repeat_layer(
        input=din, num_repeats=10, act=TanhActivation(), as_row_vector=False))
