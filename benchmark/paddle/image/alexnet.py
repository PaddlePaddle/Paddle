#!/usr/bin/env python

from paddle.trainer_config_helpers import *

height = 227
width = 227
num_class = 1000
batch_size = get_config_arg('batch_size', int, 128)

args = {'height': height, 'width': width, 'color': True, 'num_class': num_class}
define_py_data_sources2(
    "train.list", None, module="provider", obj="process", args=args)

settings(
    batch_size=batch_size,
    learning_rate=0.01 / batch_size,
    learning_method=MomentumOptimizer(0.9),
    regularization=L2Regularization(0.0005 * batch_size))

# conv1
net = data_layer('data', size=height * width * 3)
net = img_conv_layer(
    input=net,
    filter_size=11,
    num_channels=3,
    num_filters=96,
    stride=4,
    padding=1)
net = img_cmrnorm_layer(input=net, size=5, scale=0.0001, power=0.75)
net = img_pool_layer(input=net, pool_size=3, stride=2)

# conv2
net = img_conv_layer(
    input=net, filter_size=5, num_filters=256, stride=1, padding=2, groups=1)
net = img_cmrnorm_layer(input=net, size=5, scale=0.0001, power=0.75)
net = img_pool_layer(input=net, pool_size=3, stride=2)

# conv3
net = img_conv_layer(
    input=net, filter_size=3, num_filters=384, stride=1, padding=1)
# conv4
net = img_conv_layer(
    input=net, filter_size=3, num_filters=384, stride=1, padding=1, groups=1)

# conv5
net = img_conv_layer(
    input=net, filter_size=3, num_filters=256, stride=1, padding=1, groups=1)
net = img_pool_layer(input=net, pool_size=3, stride=2)

net = fc_layer(
    input=net,
    size=4096,
    act=ReluActivation(),
    layer_attr=ExtraAttr(drop_rate=0.5))
net = fc_layer(
    input=net,
    size=4096,
    act=ReluActivation(),
    layer_attr=ExtraAttr(drop_rate=0.5))
net = fc_layer(input=net, size=1000, act=SoftmaxActivation())

lab = data_layer('label', num_class)
loss = cross_entropy(input=net, label=lab)
outputs(loss)
