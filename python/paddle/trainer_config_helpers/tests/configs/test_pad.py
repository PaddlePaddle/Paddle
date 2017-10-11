from paddle.trainer_config_helpers import *

settings(batch_size=1000, learning_rate=1e-5)

data = data_layer(name='data', size=2304, height=48, width=42)

conv = img_conv_layer(
    input=data,
    filter_size=3,
    num_channels=1,
    num_filters=16,
    padding=1,
    act=LinearActivation(),
    bias_attr=True)

pool = img_pool_layer(
    input=conv, num_channels=8, pool_size=2, stride=2, pool_type=MaxPooling())

pad = pad_layer(input=pool, pad_c=[2, 3], pad_h=[1, 2], pad_w=[3, 1])

outputs(pad)
