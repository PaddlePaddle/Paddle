from paddle.trainer_config_helpers import *
settings(batch_size=1000, learning_rate=1e-4)

probs = data_layer(name='probs', size=100)

outputs(
    sampling_id_layer(input=probs),  # It seems not support training

    # It seems this layer is not correct, and should be rewrite.
    # block_expand_layer(input=probs, channel=1, block_x=1, block_y=3),
)
