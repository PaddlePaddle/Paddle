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

<<<<<<< HEAD
import unittest

import numpy as np
from test_imperative_base import new_program_scope
=======
import contextlib
import unittest
import numpy as np
import six
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.layer_helper import LayerHelper
<<<<<<< HEAD
from paddle.nn import BatchNorm
=======
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.base import to_variable
from test_imperative_base import new_program_scope
from paddle.fluid.framework import _test_eager_guard

if fluid.is_compiled_with_cuda():
    fluid.set_flags({'FLAGS_cudnn_deterministic': True})
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

batch_size = 8
train_parameters = {
    "input_size": [3, 224, 224],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "learning_strategy": {
        "name": "piecewise_decay",
        "batch_size": batch_size,
        "epochs": [30, 60, 90],
<<<<<<< HEAD
        "steps": [0.1, 0.01, 0.001, 0.0001],
=======
        "steps": [0.1, 0.01, 0.001, 0.0001]
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    },
    "batch_size": batch_size,
    "lr": 0.1,
    "total_images": 6149,
}


def optimizer_setting(params, parameter_list=None):
    ls = params["learning_strategy"]
    if ls["name"] == "piecewise_decay":
        if "total_images" not in params:
            total_images = 6149
        else:
            total_images = params["total_images"]
        # TODO(Yancey1989): using lr decay if it is ready.
<<<<<<< HEAD
        # batch_size = ls["batch_size"]
        # step = int(total_images / batch_size + 1)

        # bd = [step * e for e in ls["epochs"]]
        # base_lr = params["lr"]
        # lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
        if fluid._non_static_mode():
            optimizer = fluid.optimizer.SGD(
                learning_rate=0.01, parameter_list=parameter_list
            )
=======
        #batch_size = ls["batch_size"]
        #step = int(total_images / batch_size + 1)

        #bd = [step * e for e in ls["epochs"]]
        #base_lr = params["lr"]
        #lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
        if fluid._non_static_mode():
            optimizer = fluid.optimizer.SGD(learning_rate=0.01,
                                            parameter_list=parameter_list)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        else:
            optimizer = fluid.optimizer.SGD(learning_rate=0.01)

    return optimizer


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
            groups=groups,
            bias_attr=None,
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
                            groups=groups,
                            act=None,
                            bias_attr=None)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self._batch_norm = BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)

        return y


class SqueezeExcitation(fluid.dygraph.Layer):
<<<<<<< HEAD
    def __init__(self, num_channels, reduction_ratio):

        super().__init__()
        self._num_channels = num_channels
        self._pool = paddle.nn.AdaptiveAvgPool2D(1)
        self._squeeze = paddle.nn.Linear(
            num_channels,
            num_channels // reduction_ratio,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.05)
            ),
        )
        self.act_1 = paddle.nn.ReLU()
        self._excitation = paddle.nn.Linear(
            num_channels // reduction_ratio,
            num_channels,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.05)
            ),
        )
        self.act_2 = paddle.nn.Softmax()

    def forward(self, input):
        y = self._pool(input)
        y = paddle.reshape(y, shape=[-1, self._num_channels])
        y = self._squeeze(y)
        y = self.act_1(y)
        y = self._excitation(y)
        y = self.act_2(y)
        y = paddle.tensor.math._multiply_with_axis(x=input, y=y, axis=0)
=======

    def __init__(self, num_channels, reduction_ratio):

        super(SqueezeExcitation, self).__init__()
        self._num_channels = num_channels
        self._pool = Pool2D(pool_size=0, pool_type='avg', global_pooling=True)
        self._squeeze = Linear(
            num_channels,
            num_channels // reduction_ratio,
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=0.05)),
            act='relu')
        self._excitation = Linear(
            num_channels // reduction_ratio,
            num_channels,
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=0.05)),
            act='sigmoid')

    def forward(self, input):
        y = self._pool(input)
        y = fluid.layers.reshape(y, shape=[-1, self._num_channels])
        y = self._squeeze(y)
        y = self._excitation(y)
        y = fluid.layers.elementwise_mul(x=input, y=y, axis=0)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        return y


class BottleneckBlock(fluid.dygraph.Layer):
<<<<<<< HEAD
    def __init__(
        self,
        num_channels,
        num_filters,
        stride,
        cardinality,
        reduction_ratio,
        shortcut=True,
    ):
        super().__init__()

        self.conv0 = ConvBNLayer(
            num_channels=num_channels, num_filters=num_filters, filter_size=1
        )
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            groups=cardinality,
        )
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act='relu',
        )

        self.scale = SqueezeExcitation(
            num_channels=num_filters * 4, reduction_ratio=reduction_ratio
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
                 cardinality,
                 reduction_ratio,
                 shortcut=True):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBNLayer(num_channels=num_channels,
                                 num_filters=num_filters,
                                 filter_size=1)
        self.conv1 = ConvBNLayer(num_channels=num_filters,
                                 num_filters=num_filters,
                                 filter_size=3,
                                 stride=stride,
                                 groups=cardinality)
        self.conv2 = ConvBNLayer(num_channels=num_filters,
                                 num_filters=num_filters * 4,
                                 filter_size=1,
                                 act='relu')

        self.scale = SqueezeExcitation(num_channels=num_filters * 4,
                                       reduction_ratio=reduction_ratio)

        if not shortcut:
            self.short = ConvBNLayer(num_channels=num_channels,
                                     num_filters=num_filters * 4,
                                     filter_size=1,
                                     stride=stride)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.shortcut = shortcut

        self._num_channels_out = num_filters * 4

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)
        scale = self.scale(conv2)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

<<<<<<< HEAD
        y = paddle.add(x=short, y=scale)
=======
        y = fluid.layers.elementwise_add(x=short, y=scale)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        layer_helper = LayerHelper(self.full_name(), act='relu')
        y = layer_helper.append_activation(y)
        return y


class SeResNeXt(fluid.dygraph.Layer):
<<<<<<< HEAD
    def __init__(self, layers=50, class_dim=102):
        super().__init__()

        self.layers = layers
        supported_layers = [50, 101, 152]
        assert (
            layers in supported_layers
        ), "supported layers are {} but input layer is {}".format(
            supported_layers, layers
        )
=======

    def __init__(self, layers=50, class_dim=102):
        super(SeResNeXt, self).__init__()

        self.layers = layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        if layers == 50:
            cardinality = 32
            reduction_ratio = 16
            depth = [3, 4, 6, 3]
            num_filters = [128, 256, 512, 1024]
<<<<<<< HEAD
            self.conv0 = ConvBNLayer(
                num_channels=3,
                num_filters=64,
                filter_size=7,
                stride=2,
                act='relu',
            )
            self.pool = paddle.nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
=======
            self.conv0 = ConvBNLayer(num_channels=3,
                                     num_filters=64,
                                     filter_size=7,
                                     stride=2,
                                     act='relu')
            self.pool = Pool2D(pool_size=3,
                               pool_stride=2,
                               pool_padding=1,
                               pool_type='max')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        elif layers == 101:
            cardinality = 32
            reduction_ratio = 16
            depth = [3, 4, 23, 3]
            num_filters = [128, 256, 512, 1024]
<<<<<<< HEAD
            self.conv0 = ConvBNLayer(
                num_channels=3,
                num_filters=64,
                filter_size=7,
                stride=2,
                act='relu',
            )
            self.pool = paddle.nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
=======
            self.conv0 = ConvBNLayer(num_channels=3,
                                     num_filters=64,
                                     filter_size=7,
                                     stride=2,
                                     act='relu')
            self.pool = Pool2D(pool_size=3,
                               pool_stride=2,
                               pool_padding=1,
                               pool_type='max')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        elif layers == 152:
            cardinality = 64
            reduction_ratio = 16
            depth = [3, 8, 36, 3]
            num_filters = [128, 256, 512, 1024]
<<<<<<< HEAD
            self.conv0 = ConvBNLayer(
                num_channels=3,
                num_filters=64,
                filter_size=3,
                stride=2,
                act='relu',
            )
            self.conv1 = ConvBNLayer(
                num_channels=64,
                num_filters=64,
                filter_size=3,
                stride=2,
                act='relu',
            )
            self.conv2 = ConvBNLayer(
                num_channels=64,
                num_filters=128,
                filter_size=3,
                stride=1,
                act='relu',
            )
            self.pool = paddle.nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
=======
            self.conv0 = ConvBNLayer(num_channels=3,
                                     num_filters=64,
                                     filter_size=3,
                                     stride=2,
                                     act='relu')
            self.conv1 = ConvBNLayer(num_channels=64,
                                     num_filters=64,
                                     filter_size=3,
                                     stride=2,
                                     act='relu')
            self.conv2 = ConvBNLayer(num_channels=64,
                                     num_filters=128,
                                     filter_size=3,
                                     stride=1,
                                     act='relu')
            self.pool = Pool2D(pool_size=3,
                               pool_stride=2,
                               pool_padding=1,
                               pool_type='max')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.bottleneck_block_list = []
        num_channels = 64
        if layers == 152:
            num_channels = 128
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
                        cardinality=cardinality,
                        reduction_ratio=reduction_ratio,
                        shortcut=shortcut,
                    ),
                )
                num_channels = bottleneck_block._num_channels_out
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True
        self.pool2d_avg = paddle.nn.AdaptiveAvgPool2D(1)
        import math

=======
                    BottleneckBlock(num_channels=num_channels,
                                    num_filters=num_filters[block],
                                    stride=2 if i == 0 and block != 0 else 1,
                                    cardinality=cardinality,
                                    reduction_ratio=reduction_ratio,
                                    shortcut=shortcut))
                num_channels = bottleneck_block._num_channels_out
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True

        self.pool2d_avg = Pool2D(pool_size=7,
                                 pool_type='avg',
                                 global_pooling=True)
        import math
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        stdv = 1.0 / math.sqrt(2048 * 1.0)

        self.pool2d_avg_output = num_filters[-1] * 4 * 1 * 1

<<<<<<< HEAD
        self.out = paddle.nn.Linear(
            self.pool2d_avg_output,
            class_dim,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-stdv, stdv)
            ),
        )
        self.out_act = paddle.nn.Softmax()
=======
        self.out = Linear(
            self.pool2d_avg_output,
            class_dim,
            act='softmax',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def forward(self, inputs):
        if self.layers == 50 or self.layers == 101:
            y = self.conv0(inputs)
            y = self.pool(y)
        elif self.layers == 152:
            y = self.conv0(inputs)
            y = self.conv1(y)
            y = self.conv2(y)
            y = self.pool(y)

        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)
        y = self.pool2d_avg(y)
<<<<<<< HEAD
        y = paddle.reshape(y, shape=[-1, self.pool2d_avg_output])
        y = self.out(y)
        return self.out_act(y)


class TestImperativeResneXt(unittest.TestCase):
    def reader_decorator(self, reader):
=======
        y = fluid.layers.reshape(y, shape=[-1, self.pool2d_avg_output])
        y = self.out(y)
        return y


class TestImperativeResneXt(unittest.TestCase):

    def reader_decorator(self, reader):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def _reader_imple():
            for item in reader():
                doc = np.array(item[0]).reshape(3, 224, 224)
                label = np.array(item[1]).astype('int64').reshape(1)
                yield doc, label

        return _reader_imple

    def test_se_resnext_float32(self):
        seed = 90

        batch_size = train_parameters["batch_size"]
        batch_num = 1
        epoch_num = 1

        def run_dygraph():
            paddle.seed(seed)
            paddle.framework.random._manual_program_seed(seed)

            se_resnext = SeResNeXt()
            optimizer = optimizer_setting(
<<<<<<< HEAD
                train_parameters, parameter_list=se_resnext.parameters()
            )
=======
                train_parameters, parameter_list=se_resnext.parameters())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            np.random.seed(seed)

            batch_py_reader = fluid.io.PyReader(capacity=1)
            batch_py_reader.decorate_sample_list_generator(
<<<<<<< HEAD
                paddle.batch(
                    self.reader_decorator(
                        paddle.dataset.flowers.train(use_xmap=False)
                    ),
                    batch_size=batch_size,
                    drop_last=True,
                ),
                places=fluid.CPUPlace(),
            )
=======
                paddle.batch(self.reader_decorator(
                    paddle.dataset.flowers.train(use_xmap=False)),
                             batch_size=batch_size,
                             drop_last=True),
                places=fluid.CPUPlace())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            dy_param_init_value = {}
            for param in se_resnext.parameters():
                dy_param_init_value[param.name] = param.numpy()
            for epoch_id in range(epoch_num):
                for batch_id, data in enumerate(batch_py_reader()):

                    if batch_id >= batch_num and batch_num != -1:
                        break

                    img = data[0]
                    label = data[1]
                    label.stop_gradient = True
                    label.stop_gradient = True

                    out = se_resnext(img)
<<<<<<< HEAD
                    softmax_out = paddle.nn.functional.softmax(out)
                    loss = paddle.nn.functional.cross_entropy(
                        input=softmax_out,
                        label=label,
                        reduction='none',
                        use_softmax=False,
                    )
=======
                    softmax_out = fluid.layers.softmax(out, use_cudnn=False)
                    loss = fluid.layers.cross_entropy(input=softmax_out,
                                                      label=label)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    avg_loss = paddle.mean(x=loss)

                    dy_out = avg_loss.numpy()

                    if batch_id == 0:
                        for param in se_resnext.parameters():
                            if param.name not in dy_param_init_value:
                                dy_param_init_value[param.name] = param.numpy()
                    avg_loss.backward()

                    dy_grad_value = {}
                    for param in se_resnext.parameters():
                        if param.trainable:
                            np_array = np.array(
<<<<<<< HEAD
                                param._grad_ivar().value().get_tensor()
                            )
                            dy_grad_value[
                                param.name + core.grad_var_suffix()
                            ] = np_array
=======
                                param._grad_ivar().value().get_tensor())
                            dy_grad_value[param.name +
                                          core.grad_var_suffix()] = np_array
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

                    optimizer.minimize(avg_loss)
                    se_resnext.clear_gradients()

                    dy_param_value = {}
                    for param in se_resnext.parameters():
                        dy_param_value[param.name] = param.numpy()

<<<<<<< HEAD
                    return (
                        dy_out,
                        dy_param_init_value,
                        dy_param_value,
                        dy_grad_value,
                    )

        with fluid.dygraph.guard():
            (
                dy_out,
                dy_param_init_value,
                dy_param_value,
                dy_grad_value,
            ) = run_dygraph()

        with fluid.dygraph.guard():
            (
                eager_out,
                eager_param_init_value,
                eager_param_value,
                eager_grad_value,
            ) = run_dygraph()
=======
                    return dy_out, dy_param_init_value, dy_param_value, dy_grad_value

        with fluid.dygraph.guard():
            dy_out, dy_param_init_value, dy_param_value, dy_grad_value = run_dygraph(
            )

        with fluid.dygraph.guard():
            with _test_eager_guard():
                eager_out, eager_param_init_value, eager_param_value, eager_grad_value = run_dygraph(
                )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        with new_program_scope():
            paddle.seed(seed)
            paddle.framework.random._manual_program_seed(seed)

<<<<<<< HEAD
            exe = fluid.Executor(
                fluid.CPUPlace()
                if not core.is_compiled_with_cuda()
                else fluid.CUDAPlace(0)
            )
=======
            exe = fluid.Executor(fluid.CPUPlace(
            ) if not core.is_compiled_with_cuda() else fluid.CUDAPlace(0))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            se_resnext = SeResNeXt()
            optimizer = optimizer_setting(train_parameters)

            np.random.seed(seed)
            train_reader = paddle.batch(
                paddle.dataset.flowers.train(use_xmap=False),
                batch_size=batch_size,
<<<<<<< HEAD
                drop_last=True,
            )

            img = paddle.static.data(
                name='pixel', shape=[-1, 3, 224, 224], dtype='float32'
            )
            label = paddle.static.data(
                name='label', shape=[-1, 1], dtype='int64'
            )
            out = se_resnext(img)
            softmax_out = paddle.nn.function.softmax(out)
            loss = paddle.nn.functional.cross_entropy(
                input=softmax_out,
                label=label,
                reduction='none',
                use_softmax=False,
            )
=======
                drop_last=True)

            img = fluid.layers.data(name='pixel',
                                    shape=[3, 224, 224],
                                    dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            out = se_resnext(img)
            softmax_out = fluid.layers.softmax(out, use_cudnn=False)
            loss = fluid.layers.cross_entropy(input=softmax_out, label=label)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            avg_loss = paddle.mean(x=loss)
            optimizer.minimize(avg_loss)

            # initialize params and fetch them
            static_param_init_value = {}
            static_param_name_list = []
            static_grad_name_list = []
            for param in se_resnext.parameters():
                static_param_name_list.append(param.name)
            for param in se_resnext.parameters():
                if param.trainable:
<<<<<<< HEAD
                    static_grad_name_list.append(
                        param.name + core.grad_var_suffix()
                    )

            out = exe.run(
                fluid.default_startup_program(),
                fetch_list=static_param_name_list,
            )
=======
                    static_grad_name_list.append(param.name +
                                                 core.grad_var_suffix())

            out = exe.run(fluid.default_startup_program(),
                          fetch_list=static_param_name_list)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            for i in range(len(static_param_name_list)):
                static_param_init_value[static_param_name_list[i]] = out[i]
            for epoch_id in range(epoch_num):
                for batch_id, data in enumerate(train_reader()):
                    if batch_id >= batch_num and batch_num != -1:
                        break

<<<<<<< HEAD
                    static_x_data = np.array(
                        [x[0].reshape(3, 224, 224) for x in data]
                    ).astype('float32')
                    y_data = (
                        np.array([x[1] for x in data])
                        .astype('int64')
                        .reshape([batch_size, 1])
                    )
=======
                    static_x_data = np.array([
                        x[0].reshape(3, 224, 224) for x in data
                    ]).astype('float32')
                    y_data = np.array([x[1]
                                       for x in data]).astype('int64').reshape(
                                           [batch_size, 1])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

                    fetch_list = [avg_loss.name]
                    fetch_list.extend(static_param_name_list)
                    fetch_list.extend(static_grad_name_list)
<<<<<<< HEAD
                    out = exe.run(
                        fluid.default_main_program(),
                        feed={"pixel": static_x_data, "label": y_data},
                        fetch_list=fetch_list,
                    )
=======
                    out = exe.run(fluid.default_main_program(),
                                  feed={
                                      "pixel": static_x_data,
                                      "label": y_data
                                  },
                                  fetch_list=fetch_list)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

                    static_param_value = {}
                    static_grad_value = {}
                    static_out = out[0]
                    param_start_pos = 1
<<<<<<< HEAD
                    grad_start_pos = (
                        len(static_param_name_list) + param_start_pos
                    )
                    for i in range(
                        param_start_pos,
                        len(static_param_name_list) + param_start_pos,
                    ):
                        static_param_value[
                            static_param_name_list[i - param_start_pos]
                        ] = out[i]
                    for i in range(
                        grad_start_pos,
                        len(static_grad_name_list) + grad_start_pos,
                    ):
                        static_grad_value[
                            static_grad_name_list[i - grad_start_pos]
                        ] = out[i]
=======
                    grad_start_pos = len(
                        static_param_name_list) + param_start_pos
                    for i in range(
                            param_start_pos,
                            len(static_param_name_list) + param_start_pos):
                        static_param_value[static_param_name_list[
                            i - param_start_pos]] = out[i]
                    for i in range(grad_start_pos,
                                   len(static_grad_name_list) + grad_start_pos):
                        static_grad_value[static_grad_name_list[
                            i - grad_start_pos]] = out[i]
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        np.testing.assert_allclose(static_out, dy_out, rtol=1e-05)

        self.assertEqual(len(dy_param_init_value), len(static_param_init_value))

<<<<<<< HEAD
        for key, value in static_param_init_value.items():
            np.testing.assert_allclose(
                value, dy_param_init_value[key], rtol=1e-05
            )
=======
        for key, value in six.iteritems(static_param_init_value):
            np.testing.assert_allclose(value,
                                       dy_param_init_value[key],
                                       rtol=1e-05)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            self.assertTrue(np.isfinite(value.all()))
            self.assertFalse(np.isnan(value.any()))

        self.assertEqual(len(dy_grad_value), len(static_grad_value))

<<<<<<< HEAD
        for key, value in static_grad_value.items():
=======
        for key, value in six.iteritems(static_grad_value):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            np.testing.assert_allclose(value, dy_grad_value[key], rtol=1e-05)
            self.assertTrue(np.isfinite(value.all()))
            self.assertFalse(np.isnan(value.any()))

        self.assertEqual(len(dy_param_value), len(static_param_value))
<<<<<<< HEAD
        for key, value in static_param_value.items():
=======
        for key, value in six.iteritems(static_param_value):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            np.testing.assert_allclose(value, dy_param_value[key], rtol=1e-05)
            self.assertTrue(np.isfinite(value.all()))
            self.assertFalse(np.isnan(value.any()))

        # check eager
        np.testing.assert_allclose(static_out, eager_out, rtol=1e-05)

<<<<<<< HEAD
        self.assertEqual(
            len(eager_param_init_value), len(static_param_init_value)
        )

        for key, value in static_param_init_value.items():
            np.testing.assert_allclose(
                value, eager_param_init_value[key], rtol=1e-05
            )

        self.assertEqual(len(eager_grad_value), len(static_grad_value))

        for key, value in static_grad_value.items():
            np.testing.assert_allclose(value, eager_grad_value[key], rtol=1e-05)

        self.assertEqual(len(eager_param_value), len(static_param_value))
        for key, value in static_param_value.items():
            np.testing.assert_allclose(
                value, eager_param_value[key], rtol=1e-05
            )
=======
        self.assertEqual(len(eager_param_init_value),
                         len(static_param_init_value))

        for key, value in six.iteritems(static_param_init_value):
            np.testing.assert_allclose(value,
                                       eager_param_init_value[key],
                                       rtol=1e-05)

        self.assertEqual(len(eager_grad_value), len(static_grad_value))

        for key, value in six.iteritems(static_grad_value):
            np.testing.assert_allclose(value, eager_grad_value[key], rtol=1e-05)

        self.assertEqual(len(eager_param_value), len(static_param_value))
        for key, value in six.iteritems(static_param_value):
            np.testing.assert_allclose(value,
                                       eager_param_value[key],
                                       rtol=1e-05)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
