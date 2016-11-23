from paddle.trainer_config_helpers import *

settings(learning_rate=1e-3, batch_size=1000)

img = data_layer(name='image', size=256 * 256)

# the parse_conv in config_parse.py is not strictly accurate when filter_size
# is not square. So here set square filter_size.
img_conv = img_conv_layer(
    input=img,
    num_channels=1,
    num_filters=64,
    filter_size=(32, 32),
    padding=(1, 1),
    stride=(1, 1),
    act=LinearActivation())
img_bn = batch_norm_layer(input=img_conv, act=ReluActivation())

img_norm = img_cmrnorm_layer(input=img_bn, size=32)

img_pool = img_pool_layer(input=img_conv, pool_size=32, pool_type=MaxPooling())

outputs(img_pool, img_norm)
