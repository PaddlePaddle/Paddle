#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
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
net = data_layer('data', size=height * width * 3)
net = img_conv_layer(
    input=net,
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
net = fc_layer(input=net, size=10, act=SoftmaxActivation())

lab = data_layer('label', num_class)
loss = classification_cost(input=net, label=lab)
outputs(loss)
