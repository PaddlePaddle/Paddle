#!/usr/bin/env python
from paddle.trainer_config_helpers import *

height = 224
width = 224
num_class = 1000
batch_size = get_config_arg('batch_size', int, 64)
layer_num = get_config_arg("layer_num", int, 50)
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
    learning_rate=0.01 / batch_size,
    learning_method=MomentumOptimizer(0.9),
    regularization=L2Regularization(0.0005 * batch_size))


#######################Network Configuration #############
def conv_bn_layer(name,
                  input,
                  filter_size,
                  num_filters,
                  stride,
                  padding,
                  channels=None,
                  active_type=ReluActivation()):
    """
    A wrapper for conv layer with batch normalization layers.
    Note:
    conv layer has no activation.
    """

    tmp = img_conv_layer(
        name=name + "_conv",
        input=input,
        filter_size=filter_size,
        num_channels=channels,
        num_filters=num_filters,
        stride=stride,
        padding=padding,
        act=LinearActivation(),
        bias_attr=False)
    return batch_norm_layer(
        name=name + "_bn",
        input=tmp,
        act=active_type,
        use_global_stats=is_infer)


def bottleneck_block(name, input, num_filters1, num_filters2):
    """
    A wrapper for bottlenect building block in ResNet.
    Last conv_bn_layer has no activation.
    Addto layer has activation of relu.
    """
    last_name = conv_bn_layer(
        name=name + '_branch2a',
        input=input,
        filter_size=1,
        num_filters=num_filters1,
        stride=1,
        padding=0)
    last_name = conv_bn_layer(
        name=name + '_branch2b',
        input=last_name,
        filter_size=3,
        num_filters=num_filters1,
        stride=1,
        padding=1)
    last_name = conv_bn_layer(
        name=name + '_branch2c',
        input=last_name,
        filter_size=1,
        num_filters=num_filters2,
        stride=1,
        padding=0,
        active_type=LinearActivation())

    return addto_layer(
        name=name + "_addto", input=[input, last_name], act=ReluActivation())


def mid_projection(name, input, num_filters1, num_filters2, stride=2):
    """
    A wrapper for middile projection in ResNet.
    projection shortcuts are used for increasing dimensions,
    and other shortcuts are identity
    branch1: projection shortcuts are used for increasing
    dimensions, has no activation.
    branch2x: bottleneck building block, shortcuts are identity.
    """
    # stride = 2
    branch1 = conv_bn_layer(
        name=name + '_branch1',
        input=input,
        filter_size=1,
        num_filters=num_filters2,
        stride=stride,
        padding=0,
        active_type=LinearActivation())

    last_name = conv_bn_layer(
        name=name + '_branch2a',
        input=input,
        filter_size=1,
        num_filters=num_filters1,
        stride=stride,
        padding=0)
    last_name = conv_bn_layer(
        name=name + '_branch2b',
        input=last_name,
        filter_size=3,
        num_filters=num_filters1,
        stride=1,
        padding=1)

    last_name = conv_bn_layer(
        name=name + '_branch2c',
        input=last_name,
        filter_size=1,
        num_filters=num_filters2,
        stride=1,
        padding=0,
        active_type=LinearActivation())

    return addto_layer(
        name=name + "_addto", input=[branch1, last_name], act=ReluActivation())


img = data_layer(name='image', size=height * width * 3)


def deep_res_net(res2_num=3, res3_num=4, res4_num=6, res5_num=3):
    """
    A wrapper for 50,101,152 layers of ResNet.
    res2_num: number of blocks stacked in conv2_x
    res3_num: number of blocks stacked in conv3_x
    res4_num: number of blocks stacked in conv4_x
    res5_num: number of blocks stacked in conv5_x
    """
    # For ImageNet
    # conv1: 112x112
    tmp = conv_bn_layer(
        "conv1",
        input=img,
        filter_size=7,
        channels=3,
        num_filters=64,
        stride=2,
        padding=3)
    tmp = img_pool_layer(name="pool1", input=tmp, pool_size=3, stride=2)

    # conv2_x: 56x56
    tmp = mid_projection(
        name="res2_1", input=tmp, num_filters1=64, num_filters2=256, stride=1)
    for i in xrange(2, res2_num + 1, 1):
        tmp = bottleneck_block(
            name="res2_" + str(i), input=tmp, num_filters1=64, num_filters2=256)

    # conv3_x: 28x28
    tmp = mid_projection(
        name="res3_1", input=tmp, num_filters1=128, num_filters2=512)
    for i in xrange(2, res3_num + 1, 1):
        tmp = bottleneck_block(
            name="res3_" + str(i),
            input=tmp,
            num_filters1=128,
            num_filters2=512)

    # conv4_x: 14x14
    tmp = mid_projection(
        name="res4_1", input=tmp, num_filters1=256, num_filters2=1024)
    for i in xrange(2, res4_num + 1, 1):
        tmp = bottleneck_block(
            name="res4_" + str(i),
            input=tmp,
            num_filters1=256,
            num_filters2=1024)

    # conv5_x: 7x7
    tmp = mid_projection(
        name="res5_1", input=tmp, num_filters1=512, num_filters2=2048)
    for i in xrange(2, res5_num + 1, 1):
        tmp = bottleneck_block(
            name="res5_" + str(i),
            input=tmp,
            num_filters1=512,
            num_filters2=2048)

    tmp = img_pool_layer(
        name='avgpool',
        input=tmp,
        pool_size=7,
        stride=1,
        pool_type=AvgPooling())

    return fc_layer(input=tmp, size=num_class, act=SoftmaxActivation())


if layer_num == 50:
    resnet = deep_res_net(3, 4, 6, 3)
elif layer_num == 101:
    resnet = deep_res_net(3, 4, 23, 3)
elif layer_num == 152:
    resnet = deep_res_net(3, 8, 36, 3)
else:
    print("Wrong layer number.")

if is_infer:
    outputs(resnet)
else:
    lbl = data_layer(name="label", size=num_class)
    loss = cross_entropy(name='loss', input=resnet, label=lbl)
    outputs(loss)
