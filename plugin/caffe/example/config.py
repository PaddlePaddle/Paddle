#!/usr/bin/env python

from paddle.trainer_config_helpers import *

height = 32
width = 32
num_class = 10

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
data = data_layer('data', size=height * width * 3)
label = data_layer('label', size=1)
net = img_conv_layer(
    input=data,
    filter_size=5,
    num_channels=3,
    num_filters=32,
    stride=1,
    padding=2)
net = img_pool_layer(input=net, pool_size=3, stride=2, padding=1)

# conv2
net = img_conv_layer(
    input=net, filter_size=5, num_filters=32, stride=1, padding=2)
net = img_pool_layer(
    input=net, pool_size=3, stride=2, padding=1, pool_type=AvgPooling())

# conv3
net = img_conv_layer(
    input=net, filter_size=3, num_filters=64, stride=1, padding=1)
net = img_pool_layer(
    input=net, pool_size=3, stride=2, padding=1, pool_type=AvgPooling())

net = fc_layer(input=net, size=64, act=ReluActivation())
net = caffe_layer(
    input=net,
    act=SoftmaxActivation(),
    num_weights=2,
    prototxt="layer{type:\"InnerProduct\" inner_product_param{num_output: 10 transpose: true} }"
)
loss = cross_entropy(input=net, label=label)
outputs(loss)
