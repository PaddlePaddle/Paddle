#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
<<<<<<< HEAD
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import random
import sys
import unittest

import numpy as np
from tsm_config_utils import merge_configs, parse_config, print_configs

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.jit.api import to_static
from paddle.nn import BatchNorm, Linear
=======
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

import argparse
import math
import numpy as np
import os
import random
import sys
import time
import unittest
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import declarative, ProgramTranslator, to_variable
from paddle.fluid.dygraph.nn import Conv2D, BatchNorm, Linear, Pool2D
from paddle.fluid.layer_helper import LayerHelper
from tsm_config_utils import *
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

random.seed(0)
np.random.seed(0)


def parse_args():
    parser = argparse.ArgumentParser("Paddle Video train script")
<<<<<<< HEAD
    parser.add_argument(
        '--config',
        type=str,
        default='tsm.yaml',
        help='path to config file of model',
    )
    parser.add_argument(
        '--use_gpu',
        type=bool,
        default=fluid.is_compiled_with_cuda(),
        help='default use gpu.',
    )
=======
    parser.add_argument('--config',
                        type=str,
                        default='tsm.yaml',
                        help='path to config file of model')
    parser.add_argument('--use_gpu',
                        type=bool,
                        default=fluid.is_compiled_with_cuda(),
                        help='default use gpu.')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    args = parser.parse_args(['--config', 'tsm.yaml'])
    return args


class ConvBNLayer(fluid.dygraph.Layer):
<<<<<<< HEAD
    def __init__(
        self,
        num_channels,
        num_filters,
        filter_size,
        stride=1,
        groups=1,
        act=None,
    ):
        super().__init__()

        self._conv = paddle.nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=1,
            weight_attr=fluid.param_attr.ParamAttr(),
            bias_attr=False,
        )

        self._batch_norm = BatchNorm(
            num_filters,
            act=act,
            param_attr=fluid.param_attr.ParamAttr(),
            bias_attr=fluid.param_attr.ParamAttr(),
        )
=======

    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(num_channels=num_channels,
                            num_filters=num_filters,
                            filter_size=filter_size,
                            stride=stride,
                            padding=(filter_size - 1) // 2,
                            groups=None,
                            act=None,
                            param_attr=fluid.param_attr.ParamAttr(),
                            bias_attr=False)

        self._batch_norm = BatchNorm(num_filters,
                                     act=act,
                                     param_attr=fluid.param_attr.ParamAttr(),
                                     bias_attr=fluid.param_attr.ParamAttr())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)

        return y


class BottleneckBlock(fluid.dygraph.Layer):
<<<<<<< HEAD
    def __init__(
        self, num_channels, num_filters, stride, shortcut=True, seg_num=8
    ):
        super().__init__()

        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu',
        )
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu',
        )
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None,
        )

        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=stride,
            )
=======

    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True,
                 seg_num=8):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBNLayer(num_channels=num_channels,
                                 num_filters=num_filters,
                                 filter_size=1,
                                 act='relu')
        self.conv1 = ConvBNLayer(num_channels=num_filters,
                                 num_filters=num_filters,
                                 filter_size=3,
                                 stride=stride,
                                 act='relu')
        self.conv2 = ConvBNLayer(num_channels=num_filters,
                                 num_filters=num_filters * 4,
                                 filter_size=1,
                                 act=None)

        if not shortcut:
            self.short = ConvBNLayer(num_channels=num_channels,
                                     num_filters=num_filters * 4,
                                     filter_size=1,
                                     stride=stride)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.shortcut = shortcut
        self.seg_num = seg_num
        self._num_channels_out = int(num_filters * 4)

    def forward(self, inputs):
<<<<<<< HEAD
        shifts = paddle.nn.functional.temporal_shift(
            inputs, self.seg_num, 1.0 / 8
        )
=======
        shifts = fluid.layers.temporal_shift(inputs, self.seg_num, 1.0 / 8)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        y = self.conv0(shifts)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
<<<<<<< HEAD
        y = paddle.nn.functional.relu(paddle.add(x=short, y=conv2))
=======
        y = fluid.layers.elementwise_add(x=short, y=conv2, act="relu")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        return y


class TSM_ResNet(fluid.dygraph.Layer):
<<<<<<< HEAD
    def __init__(self, name_scope, config, mode):
        super().__init__(name_scope)
=======

    def __init__(self, name_scope, config, mode):
        super(TSM_ResNet, self).__init__(name_scope)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.layers = config.MODEL.num_layers
        self.seg_num = config.MODEL.seg_num
        self.class_dim = config.MODEL.num_classes
        self.reshape_list = [
<<<<<<< HEAD
            config.MODEL.seglen * 3,
            config[mode.upper()]['target_size'],
            config[mode.upper()]['target_size'],
=======
            config.MODEL.seglen * 3, config[mode.upper()]['target_size'],
            config[mode.upper()]['target_size']
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        ]

        if self.layers == 50:
            depth = [3, 4, 6, 3]
        else:
            raise NotImplementedError
        num_filters = [64, 128, 256, 512]

<<<<<<< HEAD
        self.conv = ConvBNLayer(
            num_channels=3, num_filters=64, filter_size=7, stride=2, act='relu'
        )
        self.pool2d_max = paddle.nn.MaxPool2D(
            kernel_size=3, stride=2, padding=1
        )
=======
        self.conv = ConvBNLayer(num_channels=3,
                                num_filters=64,
                                filter_size=7,
                                stride=2,
                                act='relu')
        self.pool2d_max = Pool2D(pool_size=3,
                                 pool_stride=2,
                                 pool_padding=1,
                                 pool_type='max')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.bottleneck_block_list = []
        num_channels = 64

        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
<<<<<<< HEAD
                    BottleneckBlock(
                        num_channels=num_channels,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        shortcut=shortcut,
                        seg_num=self.seg_num,
                    ),
                )
                num_channels = int(bottleneck_block._num_channels_out)
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True
        self.pool2d_avg = paddle.nn.AdaptiveAvgPool2D(1)
        import math

=======
                    BottleneckBlock(num_channels=num_channels,
                                    num_filters=num_filters[block],
                                    stride=2 if i == 0 and block != 0 else 1,
                                    shortcut=shortcut,
                                    seg_num=self.seg_num))
                num_channels = int(bottleneck_block._num_channels_out)
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True
        self.pool2d_avg = Pool2D(pool_size=7,
                                 pool_type='avg',
                                 global_pooling=True)

        import math
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        stdv = 1.0 / math.sqrt(2048 * 1.0)

        self.out = Linear(
            2048,
            self.class_dim,
<<<<<<< HEAD
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-stdv, stdv)
            ),
            bias_attr=paddle.ParamAttr(
                learning_rate=2.0, regularizer=paddle.regularizer.L1Decay()
            ),
        )

    @to_static
    def forward(self, inputs):
        y = paddle.reshape(inputs, [-1] + self.reshape_list)
=======
            act="softmax",
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv)),
            bias_attr=fluid.param_attr.ParamAttr(
                learning_rate=2.0, regularizer=fluid.regularizer.L2Decay(0.)))

    @declarative
    def forward(self, inputs):
        y = fluid.layers.reshape(inputs, [-1] + self.reshape_list)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        y = self.conv(y)
        y = self.pool2d_max(y)
        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)
        y = self.pool2d_avg(y)
<<<<<<< HEAD
        y = paddle.nn.functional.dropout(y, p=0.5)
        y = paddle.reshape(y, [-1, self.seg_num, y.shape[1]])
        y = paddle.mean(y, axis=1)
        y = paddle.reshape(y, shape=[-1, 2048])
        y = self.out(y)
        y = paddle.nn.functional.softmax(y)
        return y


class FakeDataReader:
=======
        y = fluid.layers.dropout(y, dropout_prob=0.5)
        y = fluid.layers.reshape(y, [-1, self.seg_num, y.shape[1]])
        y = fluid.layers.reduce_mean(y, dim=1)
        y = fluid.layers.reshape(y, shape=[-1, 2048])
        y = self.out(y)
        return y


class FakeDataReader(object):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self, mode, cfg):
        self.format = cfg.MODEL.format
        self.num_classes = cfg.MODEL.num_classes
        self.seg_num = cfg.MODEL.seg_num
        self.seglen = cfg.MODEL.seglen

        self.target_size = cfg[mode.upper()]['target_size']
<<<<<<< HEAD
        self.img_mean = (
            np.array(cfg.MODEL.image_mean).reshape([3, 1, 1]).astype(np.float32)
        )
        self.img_std = (
            np.array(cfg.MODEL.image_std).reshape([3, 1, 1]).astype(np.float32)
        )

        self.batch_size = (
            1
            if sys.platform == 'darwin' or os.name == 'nt'
            else cfg[mode.upper()]['batch_size']
        )
=======
        self.img_mean = np.array(cfg.MODEL.image_mean).reshape(
            [3, 1, 1]).astype(np.float32)
        self.img_std = np.array(cfg.MODEL.image_std).reshape([3, 1, 1]).astype(
            np.float32)

        self.batch_size = 1 if sys.platform == 'darwin' or os.name == 'nt' else cfg[
            mode.upper()]['batch_size']
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.generator_out = []
        self.total_iter = 3
        for i in range(self.total_iter):
            batch_out = []
            for j in range(self.batch_size):
                label = np.int64(random.randint(0, self.num_classes - 1))
                random_mean = self.img_mean[0][0][0]
                random_std = self.img_std[0][0][0]
<<<<<<< HEAD
                imgs = np.random.normal(
                    random_mean,
                    random_std,
                    [
                        self.seg_num,
                        self.seglen * 3,
                        self.target_size,
                        self.target_size,
                    ],
                ).astype(np.float32)
=======
                imgs = np.random.normal(random_mean, random_std, [
                    self.seg_num, self.seglen * 3, self.target_size,
                    self.target_size
                ]).astype(np.float32)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                batch_out.append((imgs, label))
            self.generator_out.append(batch_out)

    def create_reader(self):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def batch_reader():
            for i in range(self.total_iter):
                yield self.generator_out[i]

        return batch_reader


def create_optimizer(cfg, params):
    total_videos = cfg.total_videos
<<<<<<< HEAD
    batch_size = (
        1 if sys.platform == 'darwin' or os.name == 'nt' else cfg.batch_size
    )
=======
    batch_size = 1 if sys.platform == 'darwin' or os.name == 'nt' else cfg.batch_size
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    step = int(total_videos / batch_size + 1)
    bd = [e * step for e in cfg.decay_epochs]
    base_lr = cfg.learning_rate
    lr_decay = cfg.learning_rate_decay
    lr = [base_lr, base_lr * lr_decay, base_lr * lr_decay * lr_decay]
    l2_weight_decay = cfg.l2_weight_decay
    momentum = cfg.momentum

    optimizer = fluid.optimizer.Momentum(
        learning_rate=fluid.layers.piecewise_decay(boundaries=bd, values=lr),
        momentum=momentum,
        regularization=fluid.regularizer.L2Decay(l2_weight_decay),
<<<<<<< HEAD
        parameter_list=params,
    )
=======
        parameter_list=params)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    return optimizer


def train(args, fake_data_reader, to_static):
<<<<<<< HEAD
    paddle.jit.enable_to_static(to_static)
=======
    program_translator = ProgramTranslator()
    program_translator.enable(to_static)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    config = parse_config(args.config)
    train_config = merge_configs(config, 'train', vars(args))
    valid_config = merge_configs(config, 'valid', vars(args))
    print_configs(train_config, 'Train')

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()

    random.seed(0)
    np.random.seed(0)
    with fluid.dygraph.guard(place):
        paddle.seed(1000)
        paddle.framework.random._manual_program_seed(1000)

        video_model = TSM_ResNet("TSM", train_config, 'Train')

<<<<<<< HEAD
        optimizer = create_optimizer(
            train_config.TRAIN, video_model.parameters()
        )
=======
        optimizer = create_optimizer(train_config.TRAIN,
                                     video_model.parameters())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        train_reader = fake_data_reader.create_reader()

        ret = []
        for epoch in range(train_config.TRAIN.epoch):
            video_model.train()
            total_loss = 0.0
            total_acc1 = 0.0
            total_acc5 = 0.0
            total_sample = 0
            for batch_id, data in enumerate(train_reader()):
                x_data = np.array([item[0] for item in data])
                y_data = np.array([item[1] for item in data]).reshape([-1, 1])

                imgs = to_variable(x_data)
                labels = to_variable(y_data)
                labels.stop_gradient = True
                outputs = video_model(imgs)
<<<<<<< HEAD
                loss = paddle.nn.functional.cross_entropy(
                    input=outputs,
                    label=labels,
                    ignore_index=-1,
                    reduction='none',
                    use_softmax=False,
                )
                avg_loss = paddle.mean(loss)
                acc_top1 = paddle.static.accuracy(
                    input=outputs, label=labels, k=1
                )
                acc_top5 = paddle.static.accuracy(
                    input=outputs, label=labels, k=5
                )
=======
                loss = fluid.layers.cross_entropy(input=outputs,
                                                  label=labels,
                                                  ignore_index=-1)
                avg_loss = paddle.mean(loss)
                acc_top1 = fluid.layers.accuracy(input=outputs,
                                                 label=labels,
                                                 k=1)
                acc_top5 = fluid.layers.accuracy(input=outputs,
                                                 label=labels,
                                                 k=5)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

                avg_loss.backward()
                optimizer.minimize(avg_loss)
                video_model.clear_gradients()

                total_loss += avg_loss.numpy()[0]
                total_acc1 += acc_top1.numpy()[0]
                total_acc5 += acc_top5.numpy()[0]
                total_sample += 1

<<<<<<< HEAD
                print(
                    'TRAIN Epoch {}, iter {}, loss = {}, acc1 {}, acc5 {}'.format(
                        epoch,
                        batch_id,
                        avg_loss.numpy()[0],
                        acc_top1.numpy()[0],
                        acc_top5.numpy()[0],
                    )
                )
                ret.extend(
                    [
                        avg_loss.numpy()[0],
                        acc_top1.numpy()[0],
                        acc_top5.numpy()[0],
                    ]
                )

            print(
                'TRAIN End, Epoch {}, avg_loss= {}, avg_acc1= {}, avg_acc5= {}'.format(
                    epoch,
                    total_loss / total_sample,
                    total_acc1 / total_sample,
                    total_acc5 / total_sample,
                )
            )
=======
                print('TRAIN Epoch {}, iter {}, loss = {}, acc1 {}, acc5 {}'.
                      format(epoch, batch_id,
                             avg_loss.numpy()[0],
                             acc_top1.numpy()[0],
                             acc_top5.numpy()[0]))
                ret.extend([
                    avg_loss.numpy()[0],
                    acc_top1.numpy()[0],
                    acc_top5.numpy()[0]
                ])

            print(
                'TRAIN End, Epoch {}, avg_loss= {}, avg_acc1= {}, avg_acc5= {}'.
                format(epoch, total_loss / total_sample,
                       total_acc1 / total_sample, total_acc5 / total_sample))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        return ret


class TestTsm(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_dygraph_static_same_loss(self):
        if fluid.is_compiled_with_cuda():
            fluid.set_flags({"FLAGS_cudnn_deterministic": True})
        args = parse_args()
        fake_data_reader = FakeDataReader("train", parse_config(args.config))
        dygraph_loss = train(args, fake_data_reader, to_static=False)
        static_loss = train(args, fake_data_reader, to_static=True)
        np.testing.assert_allclose(dygraph_loss, static_loss, rtol=1e-05)


if __name__ == '__main__':
<<<<<<< HEAD
    unittest.main()
=======
    # switch into new eager mode
    with fluid.framework._test_eager_guard():
        unittest.main()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
