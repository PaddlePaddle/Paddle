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

import paddle
import paddle.fluid as fluid
import math
import os
from imagenet_reader import train, val

__all__ = [
    "SE_ResNeXt", "SE_ResNeXt50_32x4d", "SE_ResNeXt101_32x4d",
    "SE_ResNeXt152_32x4d", "get_model"
]

train_parameters = {
    "input_size": [3, 224, 224],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "learning_strategy": {
        "name": "piecewise_decay",
        "batch_size": 256,
        "epochs": [30, 60, 90],
        "steps": [0.1, 0.01, 0.001, 0.0001]
    }
}


class SE_ResNeXt():
    def __init__(self, layers=50, is_train=True):
        self.params = train_parameters
        self.layers = layers
        self.is_train = is_train

    def net(self, input, class_dim=1000):
        layers = self.layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)
        if layers == 50:
            cardinality = 32
            reduction_ratio = 16
            depth = [3, 4, 6, 3]
            num_filters = [128, 256, 512, 1024]

            conv = self.conv_bn_layer(
                input=input,
                num_filters=64,
                filter_size=7,
                stride=2,
                act='relu')
            conv = fluid.layers.pool2d(
                input=conv,
                pool_size=3,
                pool_stride=2,
                pool_padding=1,
                pool_type='max')
        elif layers == 101:
            cardinality = 32
            reduction_ratio = 16
            depth = [3, 4, 23, 3]
            num_filters = [128, 256, 512, 1024]

            conv = self.conv_bn_layer(
                input=input,
                num_filters=64,
                filter_size=7,
                stride=2,
                act='relu')
            conv = fluid.layers.pool2d(
                input=conv,
                pool_size=3,
                pool_stride=2,
                pool_padding=1,
                pool_type='max')
        elif layers == 152:
            cardinality = 64
            reduction_ratio = 16
            depth = [3, 8, 36, 3]
            num_filters = [128, 256, 512, 1024]

            conv = self.conv_bn_layer(
                input=input,
                num_filters=64,
                filter_size=3,
                stride=2,
                act='relu')
            conv = self.conv_bn_layer(
                input=conv, num_filters=64, filter_size=3, stride=1, act='relu')
            conv = self.conv_bn_layer(
                input=conv,
                num_filters=128,
                filter_size=3,
                stride=1,
                act='relu')
            conv = fluid.layers.pool2d(
                input=conv, pool_size=3, pool_stride=2, pool_padding=1, \
                pool_type='max')

        for block in range(len(depth)):
            for i in range(depth[block]):
                conv = self.bottleneck_block(
                    input=conv,
                    num_filters=num_filters[block],
                    stride=2 if i == 0 and block != 0 else 1,
                    cardinality=cardinality,
                    reduction_ratio=reduction_ratio)

        pool = fluid.layers.pool2d(
            input=conv, pool_size=7, pool_type='avg', global_pooling=True)
        drop = fluid.layers.dropout(x=pool, dropout_prob=0.5)
        stdv = 1.0 / math.sqrt(drop.shape[1] * 1.0)
        out = fluid.layers.fc(input=drop,
                              size=class_dim,
                              act='softmax',
                              param_attr=fluid.param_attr.ParamAttr(
                                  initializer=fluid.initializer.Uniform(-stdv,
                                                                        stdv)))
        return out

    def shortcut(self, input, ch_out, stride):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            filter_size = 1
            return self.conv_bn_layer(input, ch_out, filter_size, stride)
        else:
            return input

    def bottleneck_block(self, input, num_filters, stride, cardinality,
                         reduction_ratio):
        conv0 = self.conv_bn_layer(
            input=input, num_filters=num_filters, filter_size=1, act='relu')
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            groups=cardinality,
            act='relu')
        conv2 = self.conv_bn_layer(
            input=conv1, num_filters=num_filters * 2, filter_size=1, act=None)
        scale = self.squeeze_excitation(
            input=conv2,
            num_channels=num_filters * 2,
            reduction_ratio=reduction_ratio)

        short = self.shortcut(input, num_filters * 2, stride)

        return fluid.layers.elementwise_add(x=short, y=scale, act='relu')

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
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
        return fluid.layers.batch_norm(
            input=conv, act=act, is_test=not self.is_train)

    def squeeze_excitation(self, input, num_channels, reduction_ratio):
        pool = fluid.layers.pool2d(
            input=input, pool_size=0, pool_type='avg', global_pooling=True)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        squeeze = fluid.layers.fc(input=pool,
                                  size=num_channels / reduction_ratio,
                                  act='relu',
                                  param_attr=fluid.param_attr.ParamAttr(
                                      initializer=fluid.initializer.Uniform(
                                          -stdv, stdv)))
        stdv = 1.0 / math.sqrt(squeeze.shape[1] * 1.0)
        excitation = fluid.layers.fc(input=squeeze,
                                     size=num_channels,
                                     act='sigmoid',
                                     param_attr=fluid.param_attr.ParamAttr(
                                         initializer=fluid.initializer.Uniform(
                                             -stdv, stdv)))
        scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
        return scale


def SE_ResNeXt50_32x4d():
    model = SE_ResNeXt(layers=50)
    return model


def SE_ResNeXt101_32x4d():
    model = SE_ResNeXt(layers=101)
    return model


def SE_ResNeXt152_32x4d():
    model = SE_ResNeXt(layers=152)
    return model


def get_model(args, is_train, main_prog, startup_prog):
    model = SE_ResNeXt(layers=50)
    batched_reader = None
    pyreader = None
    trainer_count = int(os.getenv("PADDLE_TRAINERS"))
    dshape = train_parameters["input_size"]

    with fluid.program_guard(main_prog, startup_prog):
        with fluid.unique_name.guard():
            if args.use_reader_op:
                pyreader = fluid.layers.py_reader(
                    capacity=10,
                    shapes=([-1] + dshape, (-1, 1)),
                    dtypes=('float32', 'int64'),
                    name="train_reader" if is_train else "test_reader",
                    use_double_buffer=True)
                input, label = fluid.layers.read_file(pyreader)
            else:
                input = fluid.layers.data(
                    name='data', shape=dshape, dtype='float32')
                label = fluid.layers.data(
                    name='label', shape=[1], dtype='int64')

            out = model.net(input=input)
            cost = fluid.layers.cross_entropy(input=out, label=label)
            avg_cost = fluid.layers.mean(x=cost)
            acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
            acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)

            optimizer = None
            if is_train:
                total_images = 1281167 / trainer_count

                step = int(total_images / args.batch_size + 1)
                epochs = [40, 80, 100]
                bd = [step * e for e in epochs]
                base_lr = args.learning_rate
                lr = []
                lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
                optimizer = fluid.optimizer.Momentum(
                    # learning_rate=base_lr,
                    learning_rate=fluid.layers.piecewise_decay(
                        boundaries=bd, values=lr),
                    momentum=0.9,
                    regularization=fluid.regularizer.L2Decay(1e-4))
                optimizer.minimize(avg_cost)

                if args.memory_optimize:
                    fluid.memory_optimize(main_prog)

    # config readers
    if is_train:
        reader = train()
    else:
        reader = val()

    if not args.use_reader_op:
        batched_reader = paddle.batch(
            reader, batch_size=args.batch_size * args.gpus, drop_last=True)
    else:
        pyreader.decorate_paddle_reader(
            paddle.batch(
                reader, batch_size=args.batch_size))

    return avg_cost, optimizer, [acc_top1, acc_top5], batched_reader, pyreader
