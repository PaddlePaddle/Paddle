from paddle.trainer_config_helpers import *

settings(batch_size=1000, learning_rate=1e-5)

data = data_layer(name='data', size=2016, height=48, width=42)
indices = data_layer(name='indices', size=6)

scale_sub_region = scale_sub_region_layer(
    input=data, indices=indices, value=0.0)

outputs(scale_sub_region)
