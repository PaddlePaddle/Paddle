#!/usr/bin/env python
from paddle.trainer_config_helpers import *

height = 224
width = 224
num_class = 1000
batch_size = get_config_arg('batch_size', int, 64)
layer_num = get_config_arg('layer_num', int, 19)
is_infer = get_config_arg("is_infer", bool, False)
num_samples = get_config_arg('num_samples', int, 2560)

args = {
    'height': height,
    'width': width,
    'color': True,
    'num_class': num_class,
    'is_infer': is_infer,
    'num_samples': num_samples
}
define_py_data_sources2(
    "train.list" if not is_infer else None,
    "test.list" if is_infer else None,
    module="provider",
    obj="process",
    args=args)

settings(
    batch_size=batch_size,
    learning_rate=0.001 / batch_size,
    learning_method=MomentumOptimizer(0.9),
    regularization=L2Regularization(0.0005 * batch_size))

img = data_layer(name='image', size=height * width * 3)


def vgg_network(vgg_num=3):
    tmp = img_conv_group(
        input=img,
        num_channels=3,
        conv_padding=1,
        conv_num_filter=[64, 64],
        conv_filter_size=3,
        conv_act=ReluActivation(),
        pool_size=2,
        pool_stride=2,
        pool_type=MaxPooling())

    tmp = img_conv_group(
        input=tmp,
        conv_num_filter=[128, 128],
        conv_padding=1,
        conv_filter_size=3,
        conv_act=ReluActivation(),
        pool_stride=2,
        pool_type=MaxPooling(),
        pool_size=2)

    channels = []
    for i in range(vgg_num):
        channels.append(256)
    tmp = img_conv_group(
        input=tmp,
        conv_num_filter=channels,
        conv_padding=1,
        conv_filter_size=3,
        conv_act=ReluActivation(),
        pool_stride=2,
        pool_type=MaxPooling(),
        pool_size=2)
    channels = []
    for i in range(vgg_num):
        channels.append(512)
    tmp = img_conv_group(
        input=tmp,
        conv_num_filter=channels,
        conv_padding=1,
        conv_filter_size=3,
        conv_act=ReluActivation(),
        pool_stride=2,
        pool_type=MaxPooling(),
        pool_size=2)
    tmp = img_conv_group(
        input=tmp,
        conv_num_filter=channels,
        conv_padding=1,
        conv_filter_size=3,
        conv_act=ReluActivation(),
        pool_stride=2,
        pool_type=MaxPooling(),
        pool_size=2)

    tmp = fc_layer(
        input=tmp,
        size=4096,
        act=ReluActivation(),
        layer_attr=ExtraAttr(drop_rate=0.5))

    tmp = fc_layer(
        input=tmp,
        size=4096,
        act=ReluActivation(),
        layer_attr=ExtraAttr(drop_rate=0.5))

    return fc_layer(input=tmp, size=num_class, act=SoftmaxActivation())


if layer_num == 16:
    vgg = vgg_network(3)
elif layer_num == 19:
    vgg = vgg_network(4)
else:
    print("Wrong layer number.")

if is_infer:
    outputs(vgg)
else:
    lab = data_layer('label', num_class)
    loss = cross_entropy(input=vgg, label=lab)
    outputs(loss)
