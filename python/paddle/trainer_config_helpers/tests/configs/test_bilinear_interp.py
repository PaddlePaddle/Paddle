from paddle.trainer_config_helpers import *

settings(batch_size=1000, learning_rate=1e-5)

data = data_layer(name='data', size=2304)

conv = img_conv_layer(
    input=data,
    filter_size=3,
    num_channels=1,
    num_filters=16,
    padding=1,
    act=LinearActivation(),
    bias_attr=True)

bilinear = bilinear_interp_layer(input=conv, out_size_x=64, out_size_y=64)

pool = img_pool_layer(
    input=bilinear,
    num_channels=16,
    pool_size=2,
    stride=2,
    pool_type=MaxPooling())

fc = fc_layer(input=pool, size=384, bias_attr=False)

outputs(fc)
