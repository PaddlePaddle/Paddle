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

from __future__ import print_function
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import numpy as np
import os
import shutil
import unittest
import math


def conv_bn_layer(input, num_filters, filter_size, stride=1, groups=1,
                  act=None):
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=(filter_size - 1) // 2,
        groups=groups,
        act=None,
        bias_attr=False)
    return fluid.layers.batch_norm(
        input=conv,
        act=act, )


def shortcut(input, ch_out, stride, is_first):
    ch_in = input.shape[1]
    if ch_in != ch_out or stride != 1 or is_first == True:
        return conv_bn_layer(input, ch_out, 1, stride)
    else:
        return input


def bottleneck_block(input, num_filters, stride):
    conv0 = conv_bn_layer(
        input=input, num_filters=num_filters, filter_size=1, act='relu')
    conv1 = conv_bn_layer(
        input=conv0,
        num_filters=num_filters,
        filter_size=3,
        stride=stride,
        act='relu')
    conv2 = conv_bn_layer(
        input=conv1, num_filters=num_filters * 4, filter_size=1, act=None)

    short = shortcut(input, num_filters * 4, stride, is_first=False)

    return fluid.layers.elementwise_add(x=short, y=conv2, act='relu')


def basic_block(input, num_filters, stride, is_first):
    conv0 = conv_bn_layer(
        input=input,
        num_filters=num_filters,
        filter_size=3,
        act='relu',
        stride=stride)
    conv1 = conv_bn_layer(
        input=conv0, num_filters=num_filters, filter_size=3, act=None)
    short = shortcut(input, num_filters, stride, is_first)
    return fluid.layers.elementwise_add(x=short, y=conv1, act='relu')


def build_network(input, layers=50, class_dim=1000):
    supported_layers = [18, 34, 50, 101, 152]
    assert layers in supported_layers
    depth = None
    if layers == 18:
        depth = [2, 2, 2, 2]
    elif layers == 34 or layers == 50:
        depth = [3, 4, 6, 3]
    elif layers == 101:
        depth = [3, 4, 23, 3]
    elif layers == 152:
        depth = [3, 8, 36, 3]
    num_filters = [64, 128, 256, 512]
    with fluid.device_guard("cpu"):
        conv = conv_bn_layer(
            input=input, num_filters=64, filter_size=7, stride=2, act='relu')
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')
    if layers >= 50:
        for block in range(len(depth)):
            with fluid.device_guard("gpu:0"):
                for i in range(depth[block]):
                    conv = bottleneck_block(
                        input=conv,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1)

        with fluid.device_guard("gpu:0"):
            pool = fluid.layers.pool2d(
                input=conv, pool_size=7, pool_type='avg', global_pooling=True)
            stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
            out = fluid.layers.fc(
                input=pool,
                size=class_dim,
                param_attr=fluid.param_attr.ParamAttr(
                    initializer=fluid.initializer.Uniform(-stdv, stdv)))
    else:
        for block in range(len(depth)):
            with fluid.device_guard("gpu:0"):
                for i in range(depth[block]):
                    conv = basic_block(
                        input=conv,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        is_first=block == i == 0)
        with fluid.device_guard("gpu:0"):
            pool = fluid.layers.pool2d(
                input=conv, pool_size=7, pool_type='avg', global_pooling=True)
            stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
            out = fluid.layers.fc(
                input=pool,
                size=class_dim,
                param_attr=fluid.param_attr.ParamAttr(
                    initializer=fluid.initializer.Uniform(-stdv, stdv)))
    return out


class TestPipeline(unittest.TestCase):
    """  TestCases for Pipeline Training. """

    def _run(self, debug):
        main_prog = fluid.Program()
        startup_prog = fluid.Program()
        with fluid.program_guard(main_prog, startup_prog):
            with fluid.device_guard("cpu"):
                image = fluid.layers.data(
                    name="image", shape=[3, 224, 224], dtype="float32")
                label = fluid.layers.data(
                    name="label", shape=[1], dtype="int64")
                data_loader = fluid.io.DataLoader.from_generator(
                    feed_list=[image, label],
                    capacity=64,
                    use_double_buffer=True,
                    iterable=False)
                fc = build_network(image, layers=50)
            with fluid.device_guard("gpu:0"):
                out, prob = fluid.layers.softmax_with_cross_entropy(
                    logits=fc, label=label, return_softmax=True)
                loss = fluid.layers.mean(out)
                acc_top1 = fluid.layers.accuracy(input=prob, label=label, k=1)
                acc_top5 = fluid.layers.accuracy(input=prob, label=label, k=5)

            base_lr = 0.1
            passes = [30, 60, 80, 90]
            total_images = 1281167
            steps_per_pass = total_images // 128
            bd = [steps_per_pass * p for p in passes]
            lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
            lr_val = fluid.layers.piecewise_decay(boundaries=bd, values=lr)
            optimizer = fluid.optimizer.MomentumOptimizer(
                lr_val,
                momentum=0.9,
                regularization=fluid.regularizer.L2Decay(1e-4))
            optimizer = fluid.optimizer.PipelineOptimizer(
                optimizer, num_microbatches=2)
            optimizer.minimize(loss)

        def train_reader():
            for _ in range(4):
                img = np.random.random(size=[3, 224, 224]).astype('float32')
                label = np.random.random(size=[1]).astype('int64')
                yield img, label

        data_loader.set_sample_generator(train_reader, batch_size=1)
        place = fluid.CPUPlace()

        # The following dataset is only used for the 
        # interface 'train_from_dataset'.
        # And it has no actual meaning.
        dataset = fluid.DatasetFactory().create_dataset('FileInstantDataset')
        dataset.set_batch_size(1)
        dataset.set_thread(1)
        dataset.set_filelist(['/tmp/tmp_2.txt'])
        dataset.set_use_var([image, label])
        exe = fluid.Executor(place)
        exe.run(startup_prog)
        data_loader.start()
        exe.train_from_dataset(main_prog, dataset, debug=debug)

    def test_pipeline(self):
        self._run(False)
        self._run(True)

    def test_pipeline_noneoptimizer(self):
        with fluid.device_guard("gpu:0"):
            x = fluid.layers.data(
                name='x', shape=[1], dtype='int64', lod_level=0)
            y = fluid.layers.data(
                name='y', shape=[1], dtype='int64', lod_level=0)
            emb_x = layers.embedding(
                input=x,
                param_attr=fluid.ParamAttr(name="embx"),
                size=[10, 2],
                is_sparse=False)

            fc = layers.fc(input=emb_x,
                           name="fc",
                           size=1,
                           num_flatten_dims=1,
                           bias_attr=False)
            loss = layers.reduce_mean(fc)

        optimizer = fluid.optimizer.SGD(learning_rate=0.5)
        with self.assertRaises(ValueError):
            optimizer = fluid.optimizer.PipelineOptimizer(
                dict(), num_microbatches=2)


if __name__ == '__main__':
    unittest.main()
