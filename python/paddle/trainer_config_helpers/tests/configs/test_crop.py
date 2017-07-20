from paddle.trainer_config_helpers import *

settings(batch_size=1000, learning_rate=1e-5)

data = data_layer(name='data', size=2016, height=48, width=42)
refernce_data = data_layer(name='data', size=768, height=16, width=16)

conv = img_conv_layer(
    input=data,
    filter_size=3,
    num_channels=1,
    num_filters=16,
    padding=1,
    act=LinearActivation(),
    bias_attr=True)

pool = img_pool_layer(input=conv, pool_size=2, stride=2, pool_type=MaxPooling())

crop = crop_layer(input=[pool, refernce_data], axis=2)

outputs(pad)
