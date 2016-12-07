from paddle.trainer_config_helpers import *

settings(batch_size=100, learning_rate=1e-5)

data = data_layer(name='data', size=3200, height=20, width=10)

spp = spp_layer(
    input=data, pyramid_height=2, num_channels=16, pool_type=MaxPooling())

outputs(spp)
