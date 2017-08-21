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
    name='data1', size=12096 * num_channels, height=48, width=42, depth=6)

conv3d = img_conv3d_layer(
    input=data,
    name='conv3d_1',
    num_filters=16,
    num_channels=num_channels,
    filter_size=filter_size,
    filter_size_y=filter_size,
    filter_size_z=filter_size,
    stride=stride,
    stride_y=stride_y,
    stride_z=stride_z,
    padding=padding,
    padding_y=padding_y,
    padding_z=padding_z,
    groups=groups,
    bias_attr=True,
    shared_biases=True,
    trans=False,
    layer_type="conv3d",
    act=LinearActivation())

deconv3d = img_conv3d_layer(
    input=data,
    name='deconv3d_1',
    num_filters=16,
    num_channels=num_channels,
    filter_size=filter_size,
    filter_size_y=filter_size,
    filter_size_z=filter_size,
    stride=stride,
    stride_y=stride_y,
    stride_z=stride_z,
    padding=padding,
    padding_y=padding_y,
    padding_z=padding_z,
    groups=groups,
    bias_attr=True,
    shared_biases=True,
    trans=True,
    layer_type="deconv3d",
    act=LinearActivation())

data = data_layer(name="input", size=8 * 16 * 16)
conv1 = img_conv_layer(
    input=data,
    filter_size=1,
    filter_size_y=1,
    num_channels=8,
    num_filters=16,
    stride=1,
    bias_attr=False,
    act=ReluActivation(),
    layer_type="exconv")
conv2 = img_conv_layer(
    input=data,
    filter_size=1,
    filter_size_y=1,
    num_channels=8,
    num_filters=16,
    stride=1,
    bias_attr=False,
    act=ReluActivation(),
    layer_type="exconv")

concat = concat_layer(input=[conv1, conv2])

conv = img_conv_layer(
    input=data,
    filter_size=1,
    filter_size_y=1,
    num_channels=8,
    num_filters=16,
    stride=1,
    bias_attr=True,
    act=LinearActivation(),
    groups=2,
    layer_type="exconv")

outputs(concat, conv)
