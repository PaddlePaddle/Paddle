from paddle.trainer_config_helpers import *

settings(batch_size=100, learning_rate=1e-5)

data = data_layer(name='data', size=3200)

spp = spp_layer(
    input=data,
    pyramid_height=2,
    num_channels=16,
    pool_type=MaxPooling(),
    img_width=10)

outputs(spp)
