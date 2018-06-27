# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle.fluid as fluid
from parallel_executor_test_base import TestParallelExecutorBase
import unittest
import os


def squeeze_excitation(input, num_channels, reduction_ratio):
    # pool = fluid.layers.pool2d(
    #    input=input, pool_size=0, pool_type='avg', global_pooling=True)
    conv = input
    shape = conv.shape
    reshape = fluid.layers.reshape(
        x=conv, shape=[-1, shape[1], shape[2] * shape[3]])
    pool = fluid.layers.reduce_mean(input=reshape, dim=2)

    squeeze = fluid.layers.fc(input=pool,
                              size=num_channels / reduction_ratio,
                              act='relu')
    excitation = fluid.layers.fc(input=squeeze,
                                 size=num_channels,
                                 act='sigmoid')
    scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
    return scale


def conv_bn_layer(input, num_filters, filter_size, stride=1, groups=1,
                  act=None):
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=(filter_size - 1) / 2,
        groups=groups,
        act=None,
        bias_attr=False)
    return fluid.layers.batch_norm(input=conv, act=act, momentum=0.1)


def shortcut(input, ch_out, stride):
    ch_in = input.shape[1]
    if ch_in != ch_out:
        if stride == 1:
            filter_size = 1
        else:
            filter_size = 3
        return conv_bn_layer(input, ch_out, filter_size, stride)
    else:
        return input


def bottleneck_block(input, num_filters, stride, cardinality, reduction_ratio):
    # The number of first 1x1 convolutional channels for each bottleneck build block
    # was halved to reduce the compution cost.
    conv0 = conv_bn_layer(
        input=input, num_filters=num_filters, filter_size=1, act='relu')
    conv1 = conv_bn_layer(
        input=conv0,
        num_filters=num_filters * 2,
        filter_size=3,
        stride=stride,
        groups=cardinality,
        act='relu')
    conv2 = conv_bn_layer(
        input=conv1, num_filters=num_filters * 2, filter_size=1, act=None)
    scale = squeeze_excitation(
        input=conv2,
        num_channels=num_filters * 2,
        reduction_ratio=reduction_ratio)

    short = shortcut(input, num_filters * 2, stride)

    return fluid.layers.elementwise_add(x=short, y=scale, act='relu')


def SE_ResNeXt50Small(batch_size=2, use_feed=False):
    assert not use_feed, "SE_ResNeXt doesn't support feed yet"

    img = fluid.layers.fill_constant(
        shape=[batch_size, 3, 224, 224], dtype='float32', value=0.0)
    label = fluid.layers.fill_constant(
        shape=[batch_size, 1], dtype='int64', value=0.0)

    conv = conv_bn_layer(
        input=img, num_filters=16, filter_size=3, stride=2, act='relu')
    conv = conv_bn_layer(
        input=conv, num_filters=16, filter_size=3, stride=1, act='relu')
    conv = conv_bn_layer(
        input=conv, num_filters=16, filter_size=3, stride=1, act='relu')
    conv = fluid.layers.pool2d(
        input=conv, pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

    cardinality = 32
    reduction_ratio = 16
    depth = [3, 4, 6, 3]
    num_filters = [128, 256, 512, 1024]

    for block in range(len(depth)):
        for i in range(depth[block]):
            conv = bottleneck_block(
                input=conv,
                num_filters=num_filters[block],
                stride=2 if i == 0 and block != 0 else 1,
                cardinality=cardinality,
                reduction_ratio=reduction_ratio)

    shape = conv.shape
    reshape = fluid.layers.reshape(
        x=conv, shape=[-1, shape[1], shape[2] * shape[3]])
    pool = fluid.layers.reduce_mean(input=reshape, dim=2)
    dropout = fluid.layers.dropout(x=pool, dropout_prob=0.2)
    # Classifier layer:
    prediction = fluid.layers.fc(input=dropout, size=1000, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    loss = fluid.layers.mean(loss)
    return loss


class TestResnet(TestParallelExecutorBase):
    def check_resnet_convergence(self,
                                 balance_parameter_opt_between_cards,
                                 use_cuda=True,
                                 iter=20):
        os.environ['CPU_NUM'] = str(4)

        import functools
        batch_size = 2
        self.check_network_convergence(
            functools.partial(
                SE_ResNeXt50Small, batch_size=batch_size),
            iter=iter,
            batch_size=batch_size,
            use_cuda=use_cuda,
            balance_parameter_opt_between_cards=balance_parameter_opt_between_cards
        )

    def test_resnet(self):
        self.check_resnet_convergence(False, use_cuda=True)
        self.check_resnet_convergence(False, use_cuda=False, iter=5)

    def test_resnet_with_new_strategy(self):
        self.check_resnet_convergence(True, use_cuda=True)
        self.check_resnet_convergence(True, use_cuda=False, iter=5)


if __name__ == '__main__':
    unittest.main()
