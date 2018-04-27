#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import math
import unittest
import numpy
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
from paddle.fluid.initializer import init_on_cpu


def cosine_decay(learning_rate, step_each_epoch, epochs=120):
    """Applies cosine decay to the learning rate.
    lr = 0.05 * (math.cos(epoch * (math.pi / 120)) + 1)
    """
    global_step = _decay_step_counter()
    epoch = fluid.layers.floor(global_step / step_each_epoch)
    lr = learning_rate / 2.
    decayed_lr = lr * (fluid.layers.cos(epoch * (math.pi / epochs)) + 1)
    return decayed_lr


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


def SE_ResNeXt50Small():
    img = fluid.layers.data(name='img', shape=[3, 224, 224], dtype='float32')
    label = fluid.layers.data(shape=[1], dtype='int64', name='label')

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
    dropout = fluid.layers.dropout(x=pool, dropout_prob=0.2, seed=1)
    # Classifier layer:
    prediction = fluid.layers.fc(input=dropout, size=1000, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    loss = fluid.layers.mean(loss)
    return loss


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
    return fluid.layers.batch_norm(input=conv, act=act)


class ResNeXtRandomData(object):
    def __init__(self, batch_size=10):
        self.random_state = numpy.random.RandomState(seed=1)
        self.batch_size = batch_size

    def next(self):
        return {
            'img': self.random_state.uniform(
                low=-1, high=1,
                size=(self.batch_size, 3, 224, 224)).astype('float32'),
            'label': self.random_state.uniform(
                low=0, high=10, size=(self.batch_size, 1)).astype('int64')
        }


def create_unittest(network_func, data_random):
    class __cls__(unittest.TestCase):
        def test_main(self):
            startup = fluid.Program()
            startup.random_seed = 1
            main = fluid.Program()
            with fluid.program_guard(main, startup):
                loss = network_func()

            mem_opt_main = main.clone()
            fluid.memory_optimize(mem_opt_main)
            place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
            exe.run(startup)
            data = data_random()
            pe = fluid.ParallelExecutor(
                use_cuda=True,
                loss_name=loss.name,
                main_program=main,
                num_threads=10)
            for i in xrange(1000):
                pe.run(fetch_list=[], feed=next(data))
            loss_value = numpy.array(
                pe.run(fetch_list=[loss.name], feed=next(data))[0])
            del pe
            data = data_random()

            exe.run(startup)
            pe = fluid.ParallelExecutor(
                use_cuda=True,
                loss_name=loss.name,
                main_program=mem_opt_main,
                num_threads=10)
            for i in xrange(1000):
                pe.run(fetch_list=[], feed=next(data))

            loss_value_mem_opted = numpy.array(
                pe.run(fetch_list=[loss.name], feed=next(data))[0])
            self.assertAlmostEqual(loss_value[0], loss_value_mem_opted[0])

    return __cls__


TestSimpleFC = create_unittest(SE_ResNeXt50Small, ResNeXtRandomData)

if __name__ == '__main__':
    unittest.main()
