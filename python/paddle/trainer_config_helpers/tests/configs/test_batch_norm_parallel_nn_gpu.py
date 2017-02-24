from paddle.trainer_config_helpers import *

settings(batch_size=1000, learning_rate=1e-5)

data = data_layer(name='image', size=256 * 256)

img_conv = img_conv_layer(
    input=data,
    num_channels=1,
    num_filters=64,
    filter_size=(32, 32),
    padding=(1, 1),
    stride=(1, 1),
    act=LinearActivation())
bn = batch_norm_layer(
    input=img_conv, act=ReluActivation(), layer_attr=ExtraAttr(device=0))

outputs(bn)
