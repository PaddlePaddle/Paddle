from paddle.trainer_config_helpers import *

settings(batch_size=1000, learning_rate=1e-5)

num_channels = 3
filter_size = 3
filter_size_y = 3
filter_size_z = 3
stride = 2
stride_y = 2
stride_z = 2
padding = 1
padding_y = 1
padding_z = 1
groups = 1

data = data_layer(
    name='data', size=12096 * num_channels, height=48, width=42, depth=6)
# first
conv3d_1 = img_conv3d_layer(
    input=data,
    name='conv3d_1',
    num_filters=16,
    num_channels=num_channels,
    filter_size=filter_size,
    stride=stride,
    padding=padding,
    groups=groups,
    bias_attr=True,
    shared_biases=True,
    trans=False,
    layer_type="conv3d",
    act=LinearActivation())
# second
conv3d_2 = img_conv3d_layer(
    input=data,
    name='conv3d_2',
    num_filters=16,
    num_channels=num_channels,
    filter_size=[filter_size, filter_size_y, filter_size_z],
    stride=[stride, stride_y, stride_z],
    padding=[padding, padding_y, padding_z],
    groups=groups,
    bias_attr=True,
    shared_biases=True,
    trans=False,
    layer_type="conv3d",
    act=LinearActivation())
outputs(conv3d_2)
