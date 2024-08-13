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

import unittest

import numpy as np
from test_imperative_base import new_program_scope
from utils import DyGraphProgramDescTracerTestHelper

import paddle
from paddle import base
from paddle.autograd.backward_utils import ValueDict
from paddle.base import core
from paddle.nn import Linear


def create_parameter_mapping(startup_program, main_program):
    startup_params = {}
    main_params = {}
    parameter_mapping = ValueDict()
    for op in startup_program.global_block().ops:
        if op.name() == "builtin.set_parameter":
            name = op.attrs()["parameter_name"]
            param = op.operand(0).source()
            startup_params[name] = param

    for op in main_program.global_block().ops:
        if op.name() == "builtin.parameter":
            name = op.attrs()["parameter_name"]
            param = op.result(0)
            main_params[name] = param

    assert len(startup_params) == len(main_params)
    for name, startup_param in startup_params.items():
        assert name in main_params
        main_param = main_params[name]
        parameter_mapping[main_param] = startup_param
    return parameter_mapping


class SimpleImgConvPool(paddle.nn.Layer):
    def __init__(
        self,
        num_channels,
        num_filters,
        filter_size,
        pool_size,
        pool_stride,
        pool_padding=0,
        pool_type='max',
        global_pooling=False,
        conv_stride=1,
        conv_padding=0,
        conv_dilation=1,
        conv_groups=1,
        act=None,
        use_cudnn=False,
        param_attr=None,
        bias_attr=None,
    ):
        super().__init__()

        self._conv2d = paddle.nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=conv_stride,
            padding=conv_padding,
            dilation=conv_dilation,
            groups=conv_groups,
            weight_attr=None,
            bias_attr=None,
        )
        self._pool2d = paddle.nn.MaxPool2D(
            kernel_size=pool_size,
            stride=pool_stride,
            padding=pool_padding,
        )

    def forward(self, inputs):
        x = self._conv2d(inputs)
        x = self._pool2d(x)
        return x


class MNIST(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

        self._simple_img_conv_pool_1 = SimpleImgConvPool(
            1, 20, 5, 2, 2, act="relu"
        )

        self._simple_img_conv_pool_2 = SimpleImgConvPool(
            20, 50, 5, 2, 2, act="relu"
        )

        self.pool_2_shape = 50 * 4 * 4
        SIZE = 10
        scale = (2.0 / (self.pool_2_shape**2 * SIZE)) ** 0.5
        self._fc = Linear(
            self.pool_2_shape,
            10,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(mean=0.0, std=scale)
            ),
        )

    def forward(self, inputs):
        x = self._simple_img_conv_pool_1(inputs)
        x = self._simple_img_conv_pool_2(x)
        x = paddle.reshape(x, shape=[-1, self.pool_2_shape])
        x = self._fc(x)
        x = paddle.nn.functional.softmax(x)
        return x


class TestImperativeMnist(unittest.TestCase):
    def reader_decorator(self, reader):
        def _reader_simple():
            for item in reader():
                image = np.array(item[0]).reshape(1, 28, 28)
                label = np.array(item[1]).astype('int64').reshape(1)
                yield image, label

        return _reader_simple

    def test_mnist_float32(self):
        seed = 90
        epoch_num = 1
        batch_size = 128
        batch_num = 50

        traced_layer = None

        with base.dygraph.guard():
            paddle.seed(seed)

            mnist = MNIST()
            sgd = paddle.optimizer.SGD(
                learning_rate=1e-3, parameters=mnist.parameters()
            )

            batch_py_reader = base.io.PyReader(capacity=1)
            batch_py_reader.decorate_sample_list_generator(
                paddle.batch(
                    self.reader_decorator(paddle.dataset.mnist.train()),
                    batch_size=batch_size,
                    drop_last=True,
                ),
                places=base.CPUPlace(),
            )

            mnist.train()
            dy_param_init_value = {}

            helper = DyGraphProgramDescTracerTestHelper(self)
            program = None
            for epoch in range(epoch_num):
                for batch_id, data in enumerate(batch_py_reader()):
                    if batch_id >= batch_num:
                        break
                    img = data[0]
                    dy_x_data = img.numpy()
                    label = data[1]
                    label.stop_gradient = True
                    cost = mnist(img)

                    if traced_layer is not None:
                        cost_static = traced_layer([img])
                        helper.assertEachVar(cost, cost_static)

                    loss = paddle.nn.functional.cross_entropy(
                        cost, label, reduction='none', use_softmax=False
                    )
                    avg_loss = paddle.mean(loss)

                    dy_out = avg_loss.numpy()

                    if epoch == 0 and batch_id == 0:
                        for param in mnist.parameters():
                            dy_param_init_value[param.name] = param.numpy()

                    avg_loss.backward()
                    sgd.minimize(avg_loss)
                    mnist.clear_gradients()

                    dy_param_value = {}
                    for param in mnist.parameters():
                        dy_param_value[param.name] = param.numpy()

        with new_program_scope():
            paddle.seed(seed)

            exe = base.Executor(
                base.CPUPlace()
                if not core.is_compiled_with_cuda()
                else base.CUDAPlace(0)
            )

            mnist = MNIST()
            sgd = paddle.optimizer.SGD(learning_rate=1e-3)
            train_reader = paddle.batch(
                paddle.dataset.mnist.train(),
                batch_size=batch_size,
                drop_last=True,
            )

            img = paddle.static.data(
                name='pixel', shape=[-1, 1, 28, 28], dtype='float32'
            )
            label = paddle.static.data(
                name='label', shape=[-1, 1], dtype='int64'
            )
            cost = mnist(img)
            loss = paddle.nn.functional.cross_entropy(
                cost, label, reduction='none', use_softmax=False
            )
            avg_loss = paddle.mean(loss)
            sgd.minimize(avg_loss)

            # initialize params and fetch them
            static_param_init_value = {}
            static_param_name_list = []
            static_params = []
            for param in mnist.parameters():
                static_param_name_list.append(param.name)
                static_params.append(param)

            if paddle.framework.use_pir_api():
                parameter_mapping = create_parameter_mapping(
                    paddle.static.default_startup_program(),
                    paddle.static.default_main_program(),
                )
                startup_params = [
                    parameter_mapping[param] for param in static_params
                ]
            else:
                startup_params = static_params

            out = exe.run(
                paddle.static.default_startup_program(),
                fetch_list=startup_params,
            )

            for i in range(len(static_params)):
                param_name = static_param_name_list[i]
                static_param_init_value[param_name] = out[i]

            for epoch in range(epoch_num):
                for batch_id, data in enumerate(train_reader()):
                    if batch_id >= batch_num:
                        break
                    static_x_data = np.array(
                        [x[0].reshape(1, 28, 28) for x in data]
                    ).astype('float32')
                    y_data = (
                        np.array([x[1] for x in data])
                        .astype('int64')
                        .reshape([batch_size, 1])
                    )

                    fetch_list = [avg_loss]
                    fetch_list.extend(static_params)

                    if traced_layer is not None:
                        traced_layer([static_x_data])

                    out = exe.run(
                        base.default_main_program(),
                        feed={"pixel": static_x_data, "label": y_data},
                        fetch_list=fetch_list,
                    )

                    static_param_value = {}
                    static_out = out[0]
                    for i in range(1, len(out)):
                        static_param_value[static_param_name_list[i - 1]] = out[
                            i
                        ]

        np.testing.assert_allclose(
            dy_x_data.all(), static_x_data.all(), rtol=1e-05
        )

        for key, value in static_param_init_value.items():
            np.testing.assert_allclose(
                value, dy_param_init_value[key], rtol=1e-05
            )

        np.testing.assert_allclose(static_out, dy_out, rtol=1e-05)

        for key, value in static_param_value.items():
            np.testing.assert_allclose(
                value, dy_param_value[key], rtol=1e-05, atol=1e-05
            )


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
