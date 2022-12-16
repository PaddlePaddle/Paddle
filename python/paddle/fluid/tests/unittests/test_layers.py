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

import contextlib
import inspect
import unittest

import numpy as np
from decorator_helper import prog_scope
from test_imperative_base import new_program_scope

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.nets as nets
import paddle.nn.functional as F
from paddle.fluid import core
from paddle.fluid.dygraph import base, to_variable
from paddle.fluid.framework import (
    Program,
    _test_eager_guard,
    default_main_program,
    program_guard,
)
from paddle.tensor import random


class LayerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed = 111

    @classmethod
    def tearDownClass(cls):
        pass

    def _get_place(self, force_to_use_cpu=False):
        # this option for ops that only have cpu kernel
        if force_to_use_cpu:
            return core.CPUPlace()
        else:
            if core.is_compiled_with_cuda():
                return core.CUDAPlace(0)
            return core.CPUPlace()

    @contextlib.contextmanager
    def static_graph(self):
        with new_program_scope():
            paddle.seed(self.seed)
            paddle.framework.random._manual_program_seed(self.seed)
            yield

    def get_static_graph_result(
        self, feed, fetch_list, with_lod=False, force_to_use_cpu=False
    ):
        exe = fluid.Executor(self._get_place(force_to_use_cpu))
        exe.run(fluid.default_startup_program())
        return exe.run(
            fluid.default_main_program(),
            feed=feed,
            fetch_list=fetch_list,
            return_numpy=(not with_lod),
        )

    @contextlib.contextmanager
    def dynamic_graph(self, force_to_use_cpu=False):
        with fluid.dygraph.guard(
            self._get_place(force_to_use_cpu=force_to_use_cpu)
        ):
            paddle.seed(self.seed)
            paddle.framework.random._manual_program_seed(self.seed)
            yield


class TestLayer(LayerTest):
    def test_custom_layer_with_kwargs(self):
        class CustomLayer(fluid.Layer):
            def __init__(self, input_size, linear1_size=4):
                super().__init__()
                self.linear1 = paddle.nn.Linear(
                    input_size, linear1_size, bias_attr=False
                )
                self.linear2 = paddle.nn.Linear(
                    linear1_size, 1, bias_attr=False
                )

            def forward(self, x, do_linear2=False):
                ret = self.linear1(x)
                if do_linear2:
                    ret = self.linear2(ret)
                return ret

        with self.dynamic_graph():
            with _test_eager_guard():
                inp = np.ones([3, 3], dtype='float32')
                x = base.to_variable(inp)
                custom = CustomLayer(input_size=3, linear1_size=2)
                ret = custom(x, do_linear2=False)
                np.testing.assert_array_equal(ret.numpy().shape, [3, 2])
                ret = custom(x, do_linear2=True)
                np.testing.assert_array_equal(ret.numpy().shape, [3, 1])
            inp = np.ones([3, 3], dtype='float32')
            x = base.to_variable(inp)
            custom = CustomLayer(input_size=3, linear1_size=2)
            ret = custom(x, do_linear2=False)
            np.testing.assert_array_equal(ret.numpy().shape, [3, 2])
            ret = custom(x, do_linear2=True)
            np.testing.assert_array_equal(ret.numpy().shape, [3, 1])

    def test_dropout(self):
        inp = np.ones([3, 32, 32], dtype='float32')
        with self.static_graph():
            t = layers.data(
                name='data',
                shape=[3, 32, 32],
                dtype='float32',
                append_batch_size=False,
            )
            dropout = paddle.nn.Dropout(p=0.35)
            ret = dropout(t)
            ret2 = paddle.nn.functional.dropout(t, p=0.35)
            static_ret, static_ret2 = self.get_static_graph_result(
                feed={'data': inp}, fetch_list=[ret, ret2]
            )
        with self.dynamic_graph():
            with _test_eager_guard():
                t = base.to_variable(inp)
                dropout = paddle.nn.Dropout(p=0.35)
                dy_eager_ret = dropout(t)
                dy_eager_ret2 = paddle.nn.functional.dropout(t, p=0.35)
                dy_eager_ret_value = dy_eager_ret.numpy()
                dy_eager_ret2_value = dy_eager_ret2.numpy()

            t = base.to_variable(inp)
            dropout = paddle.nn.Dropout(p=0.35)
            dy_ret = dropout(t)
            dy_ret2 = paddle.nn.functional.dropout(t, p=0.35)
            dy_ret_value = dy_ret.numpy()
            dy_ret2_value = dy_ret2.numpy()

        np.testing.assert_array_equal(dy_eager_ret_value, dy_eager_ret2_value)
        np.testing.assert_array_equal(static_ret, dy_eager_ret_value)

        np.testing.assert_array_equal(static_ret, static_ret2)
        np.testing.assert_array_equal(dy_ret_value, dy_ret2_value)
        np.testing.assert_array_equal(static_ret, dy_ret_value)

    def test_linear(self):
        inp = np.ones([3, 32, 32], dtype='float32')
        with self.static_graph():
            t = layers.data(
                name='data',
                shape=[3, 32, 32],
                dtype='float32',
                append_batch_size=False,
            )
            linear = paddle.nn.Linear(
                32, 4, bias_attr=fluid.initializer.ConstantInitializer(value=1)
            )
            ret = linear(t)
            static_ret = self.get_static_graph_result(
                feed={'data': inp}, fetch_list=[ret]
            )[0]
        with self.dynamic_graph():
            with _test_eager_guard():
                t = base.to_variable(inp)
                linear = paddle.nn.Linear(
                    32,
                    4,
                    bias_attr=fluid.initializer.ConstantInitializer(value=1),
                )
                dy_eager_ret = linear(t)
                dy_eager_ret_value = dy_eager_ret.numpy()

            t = base.to_variable(inp)
            linear = paddle.nn.Linear(
                32, 4, bias_attr=fluid.initializer.ConstantInitializer(value=1)
            )
            dy_ret = linear(t)
            dy_ret_value = dy_ret.numpy()

        np.testing.assert_array_equal(static_ret, dy_eager_ret_value)
        np.testing.assert_array_equal(static_ret, dy_ret_value)

        with self.static_graph():

            # the input of Linear must be Variable.
            def test_Variable():
                inp = np.ones([3, 32, 32], dtype='float32')
                linear = paddle.nn.Linear(
                    32,
                    4,
                    bias_attr=fluid.initializer.ConstantInitializer(value=1),
                )
                linear_ret1 = linear(inp)

            self.assertRaises(TypeError, test_Variable)

            # the input dtype of Linear must be float16 or float32 or float64
            # float16 only can be set on GPU place
            def test_type():
                inp = np.ones([3, 32, 32], dtype='int32')
                linear = paddle.nn.Linear(
                    32,
                    4,
                    bias_attr=fluid.initializer.ConstantInitializer(value=1),
                )
                linear_ret2 = linear(inp)

            self.assertRaises(TypeError, test_type)

    def test_cvm(self):
        inp = np.ones([10, 10], dtype='float32')
        arr = [[0.6931472, -1.904654e-09, 1, 1, 1, 1, 1, 1, 1, 1]] * 10
        cvm1 = np.array(arr, dtype='float32')
        cvm2 = np.ones([10, 8], dtype='float32')
        show_clk = np.ones([10, 2], dtype='float32')
        with self.static_graph():
            x = paddle.static.data(
                name='data',
                shape=[10, 10],
                dtype='float32',
            )
            u = paddle.static.data(
                name='show_click',
                shape=[10, 2],
                dtype='float32',
            )
            no_cvm = paddle.static.nn.continuous_value_model(x, u, True)
            static_ret1 = self.get_static_graph_result(
                feed={'data': inp, 'show_click': show_clk},
                fetch_list=[no_cvm],
            )[0]
        with self.static_graph():
            x = paddle.static.data(
                name='data',
                shape=[10, 10],
                dtype='float32',
            )
            u = paddle.static.data(
                name='show_click',
                shape=[10, 2],
                dtype='float32',
            )
            cvm = paddle.static.nn.continuous_value_model(x, u, False)
            static_ret2 = self.get_static_graph_result(
                feed={'data': inp, 'show_click': show_clk}, fetch_list=[cvm]
            )[0]
        np.testing.assert_allclose(static_ret1, cvm1, rtol=1e-5, atol=1e-06)
        np.testing.assert_allclose(static_ret2, cvm2, rtol=1e-5, atol=1e-06)

    def test_Flatten(self):
        inp = np.ones([3, 4, 4, 5], dtype='float32')
        with self.static_graph():
            t = layers.data(
                name='data',
                shape=[3, 4, 4, 5],
                dtype='float32',
                append_batch_size=False,
            )
            flatten = paddle.nn.Flatten()
            ret = flatten(t)
            static_ret = self.get_static_graph_result(
                feed={'data': inp}, fetch_list=[ret]
            )[0]
        with self.dynamic_graph():
            with _test_eager_guard():
                t = base.to_variable(inp)
                flatten = paddle.nn.Flatten()
                dy_eager_ret = flatten(t)
                dy_eager_ret_value = dy_eager_ret.numpy()

            t = base.to_variable(inp)
            flatten = paddle.nn.Flatten()
            dy_ret = flatten(t)
            dy_ret_value = dy_ret.numpy()

        np.testing.assert_array_equal(static_ret, dy_eager_ret_value)
        np.testing.assert_array_equal(static_ret, dy_ret_value)

        with self.static_graph():

            # the input of Linear must be Variable.
            def test_Variable():
                inp = np.ones([3, 32, 32], dtype='float32')
                linear = paddle.nn.Linear(
                    32,
                    4,
                    bias_attr=fluid.initializer.ConstantInitializer(value=1),
                )
                linear_ret1 = linear(inp)

            self.assertRaises(TypeError, test_Variable)

            # the input dtype of Linear must be float16 or float32 or float64
            # float16 only can be set on GPU place
            def test_type():
                inp = np.ones([3, 32, 32], dtype='int32')
                linear = paddle.nn.Linear(
                    32,
                    4,
                    bias_attr=fluid.initializer.ConstantInitializer(value=1),
                )
                linear_ret2 = linear(inp)

            self.assertRaises(TypeError, test_type)

    def test_SyncBatchNorm(self):
        if core.is_compiled_with_cuda():
            with self.static_graph():
                t = layers.data(name='t', shape=[-1, 3, 5, 5], dtype='float32')
                my_sync_bn = paddle.nn.SyncBatchNorm(3)
                ret = my_sync_bn(t)
                static_ret = self.get_static_graph_result(
                    feed={'t': np.ones([3, 3, 5, 5], dtype='float32')},
                    fetch_list=[ret],
                )[0]

            with self.dynamic_graph():
                with _test_eager_guard():
                    t = np.ones([3, 3, 5, 5], dtype='float32')
                    my_syncbn = paddle.nn.SyncBatchNorm(3)
                    dy_eager_ret = my_syncbn(base.to_variable(t))
                    dy_eager_ret_value = dy_eager_ret.numpy()

                t = np.ones([3, 3, 5, 5], dtype='float32')
                my_syncbn = paddle.nn.SyncBatchNorm(3)
                dy_ret = my_syncbn(base.to_variable(t))
                dy_ret_value = dy_ret.numpy()
            np.testing.assert_array_equal(static_ret, dy_ret_value)
            np.testing.assert_array_equal(static_ret, dy_eager_ret_value)

    def test_relu(self):
        with self.static_graph():
            t = layers.data(name='t', shape=[3, 3], dtype='float32')
            ret = F.relu(t)
            static_ret = self.get_static_graph_result(
                feed={'t': np.ones([3, 3], dtype='float32')}, fetch_list=[ret]
            )[0]

        with self.dynamic_graph():
            with _test_eager_guard():
                t = np.ones([3, 3], dtype='float32')
                dy_eager_ret = F.relu(base.to_variable(t))
                dy_eager_ret_value = dy_eager_ret.numpy()

            t = np.ones([3, 3], dtype='float32')
            dy_ret = F.relu(base.to_variable(t))
            dy_ret_value = dy_ret.numpy()

        np.testing.assert_allclose(static_ret, dy_ret_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret, dy_eager_ret_value, rtol=1e-05)

    def test_matmul(self):
        with self.static_graph():
            t = layers.data(name='t', shape=[3, 3], dtype='float32')
            t2 = layers.data(name='t2', shape=[3, 3], dtype='float32')
            ret = paddle.matmul(t, t2)
            static_ret = self.get_static_graph_result(
                feed={
                    't': np.ones([3, 3], dtype='float32'),
                    't2': np.ones([3, 3], dtype='float32'),
                },
                fetch_list=[ret],
            )[0]

        with self.dynamic_graph():
            with _test_eager_guard():
                t = np.ones([3, 3], dtype='float32')
                t2 = np.ones([3, 3], dtype='float32')
                dy_eager_ret = paddle.matmul(
                    base.to_variable(t), base.to_variable(t2)
                )
                dy_eager_ret_value = dy_eager_ret.numpy()

            t = np.ones([3, 3], dtype='float32')
            t2 = np.ones([3, 3], dtype='float32')
            dy_ret = paddle.matmul(base.to_variable(t), base.to_variable(t2))
            dy_ret_value = dy_ret.numpy()

        np.testing.assert_allclose(static_ret, dy_ret_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret, dy_eager_ret_value, rtol=1e-05)

    def test_elementwise_math(self):
        n = np.ones([3, 3], dtype='float32')
        n2 = np.ones([3, 3], dtype='float32') * 1.1
        n3 = np.ones([3, 3], dtype='float32') * 2
        n4 = np.ones([3, 3], dtype='float32') * 3
        n5 = np.ones([3, 3], dtype='float32') * 4
        n6 = np.ones([3, 3], dtype='float32') * 5

        with self.static_graph():
            t = layers.data(name='t', shape=[3, 3], dtype='float32')
            t2 = layers.data(name='t2', shape=[3, 3], dtype='float32')
            t3 = layers.data(name='t3', shape=[3, 3], dtype='float32')
            t4 = layers.data(name='t4', shape=[3, 3], dtype='float32')
            t5 = layers.data(name='t5', shape=[3, 3], dtype='float32')
            t6 = layers.data(name='t6', shape=[3, 3], dtype='float32')

            ret = paddle.add(t, t2)
            ret = paddle.pow(ret, t3)
            ret = paddle.divide(ret, t4)
            ret = paddle.subtract(ret, t5)
            ret = paddle.multiply(ret, t6)

            static_ret = self.get_static_graph_result(
                feed={'t': n, 't2': n2, 't3': n3, 't4': n4, 't5': n5, 't6': n6},
                fetch_list=[ret],
            )[0]

        with self.dynamic_graph():
            with _test_eager_guard():
                ret = paddle.add(to_variable(n), to_variable(n2))
                ret = paddle.pow(ret, to_variable(n3))
                ret = paddle.divide(ret, to_variable(n4))
                ret = paddle.subtract(ret, to_variable(n5))
                dy_eager_ret = paddle.multiply(ret, to_variable(n6))
                dy_eager_ret_value = dy_eager_ret.numpy()

            ret = paddle.add(to_variable(n), to_variable(n2))
            ret = paddle.pow(ret, to_variable(n3))
            ret = paddle.divide(ret, to_variable(n4))
            ret = paddle.subtract(ret, to_variable(n5))
            dy_ret = paddle.multiply(ret, to_variable(n6))
            dy_ret_value = dy_ret.numpy()

        np.testing.assert_allclose(static_ret, dy_ret_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret, dy_eager_ret_value, rtol=1e-05)

    def test_elementwise_minmax(self):
        n = np.ones([3, 3], dtype='float32')
        n2 = np.ones([3, 3], dtype='float32') * 2

        with self.dynamic_graph():
            with _test_eager_guard():
                min_eager_ret = paddle.minimum(to_variable(n), to_variable(n2))
                max_eager_ret = paddle.maximum(to_variable(n), to_variable(n2))
                min_eager_ret_value = min_eager_ret.numpy()
                max_eager_ret_value = max_eager_ret.numpy()

            min_ret = paddle.minimum(to_variable(n), to_variable(n2))
            max_ret = paddle.maximum(to_variable(n), to_variable(n2))
            min_ret_value = min_ret.numpy()
            max_ret_value = max_ret.numpy()

        np.testing.assert_allclose(n, min_ret_value, rtol=1e-05)
        np.testing.assert_allclose(n2, max_ret_value, rtol=1e-05)
        np.testing.assert_allclose(n, min_eager_ret_value, rtol=1e-05)
        np.testing.assert_allclose(n2, max_eager_ret_value, rtol=1e-05)

    def test_conv2d_transpose(self):
        inp_np = np.arange(0, 24).reshape([2, 3, 2, 2]).astype('float32')
        with self.static_graph():
            img = layers.data(name='pixel', shape=[3, 2, 2], dtype='float32')
            out = paddle.static.nn.conv2d_transpose(
                input=img,
                num_filters=10,
                filter_size=27,
                act='sigmoid',
                bias_attr=fluid.initializer.ConstantInitializer(value=1),
            )
            static_rlt = self.get_static_graph_result(
                feed={'pixel': inp_np}, fetch_list=[out]
            )[0]
        with self.static_graph():
            img = layers.data(name='pixel', shape=[3, 2, 2], dtype='float32')
            conv2d_transpose = paddle.nn.Conv2DTranspose(
                3,
                10,
                27,
                bias_attr=fluid.initializer.ConstantInitializer(value=1),
            )
            out = conv2d_transpose(img)
            out = paddle.nn.functional.sigmoid(out)
            static_rlt2 = self.get_static_graph_result(
                feed={'pixel': inp_np}, fetch_list=[out]
            )[0]
        with self.dynamic_graph():
            with _test_eager_guard():
                conv2d_transpose = paddle.nn.Conv2DTranspose(
                    3,
                    10,
                    27,
                    bias_attr=fluid.initializer.ConstantInitializer(value=1),
                )
                dy_eager_rlt = conv2d_transpose(base.to_variable(inp_np))
                dy_eager_rlt = paddle.nn.functional.sigmoid(dy_eager_rlt)
                dy_eager_rlt_value = dy_eager_rlt.numpy()

            conv2d_transpose = paddle.nn.Conv2DTranspose(
                3,
                10,
                27,
                bias_attr=fluid.initializer.ConstantInitializer(value=1),
            )
            dy_rlt = conv2d_transpose(base.to_variable(inp_np))
            dy_rlt = paddle.nn.functional.sigmoid(dy_rlt)
            dy_rlt_value = dy_rlt.numpy()
        np.testing.assert_allclose(static_rlt2, static_rlt, rtol=1e-05)
        np.testing.assert_allclose(dy_rlt_value, static_rlt2, rtol=1e-05)
        np.testing.assert_allclose(dy_eager_rlt_value, static_rlt2, rtol=1e-05)

        with self.dynamic_graph():
            with _test_eager_guard():
                images = np.ones([2, 3, 5, 5], dtype='float32')
                custom_weight = np.random.randn(3, 3, 2, 2).astype("float32")
                weight_attr = fluid.ParamAttr(
                    initializer=fluid.initializer.NumpyArrayInitializer(
                        custom_weight
                    )
                )
                conv2d1 = paddle.nn.Conv2DTranspose(3, 3, [2, 2])
                conv2d2 = paddle.nn.Conv2DTranspose(
                    3,
                    3,
                    [2, 2],
                    weight_attr=weight_attr,
                )
                dy_ret1 = conv2d1(base.to_variable(images))
                dy_ret2 = conv2d2(base.to_variable(images))
                self.assertFalse(
                    np.array_equal(dy_ret1.numpy(), dy_ret2.numpy())
                )

                conv2d1_weight_np = conv2d1.weight.numpy()
                conv2d1_bias = conv2d1.bias
                self.assertFalse(
                    np.array_equal(conv2d1_weight_np, conv2d2.weight.numpy())
                )
                conv2d2.weight.set_value(conv2d1_weight_np)
                np.testing.assert_array_equal(
                    conv2d1_weight_np, conv2d2.weight.numpy()
                )
                conv2d2.bias.set_value(conv2d1_bias)
                dy_ret1 = conv2d1(base.to_variable(images))
                dy_ret2 = conv2d2(base.to_variable(images))
                np.testing.assert_array_equal(dy_ret1.numpy(), dy_ret2.numpy())

                conv2d2.weight = conv2d1.weight
                conv2d2.bias = conv2d1.bias
                np.testing.assert_array_equal(
                    conv2d1.weight.numpy(), conv2d2.weight.numpy()
                )
                np.testing.assert_array_equal(
                    conv2d1.bias.numpy(), conv2d2.bias.numpy()
                )

            images = np.ones([2, 3, 5, 5], dtype='float32')
            custom_weight = np.random.randn(3, 3, 2, 2).astype("float32")
            weight_attr = fluid.ParamAttr(
                initializer=fluid.initializer.NumpyArrayInitializer(
                    custom_weight
                )
            )
            conv2d1 = paddle.nn.Conv2DTranspose(3, 3, [2, 2])
            conv2d2 = paddle.nn.Conv2DTranspose(
                3,
                3,
                [2, 2],
                weight_attr=weight_attr,
            )
            dy_ret1 = conv2d1(base.to_variable(images))
            dy_ret2 = conv2d2(base.to_variable(images))
            self.assertFalse(np.array_equal(dy_ret1.numpy(), dy_ret2.numpy()))

            conv2d1_weight_np = conv2d1.weight.numpy()
            conv2d1_bias = conv2d1.bias
            self.assertFalse(
                np.array_equal(conv2d1_weight_np, conv2d2.weight.numpy())
            )
            conv2d2.weight.set_value(conv2d1_weight_np)
            np.testing.assert_array_equal(
                conv2d1_weight_np, conv2d2.weight.numpy()
            )
            conv2d2.bias.set_value(conv2d1_bias)
            dy_ret1 = conv2d1(base.to_variable(images))
            dy_ret2 = conv2d2(base.to_variable(images))
            np.testing.assert_array_equal(dy_ret1.numpy(), dy_ret2.numpy())

            conv2d2.weight = conv2d1.weight
            conv2d2.bias = conv2d1.bias
            np.testing.assert_array_equal(
                conv2d1.weight.numpy(), conv2d2.weight.numpy()
            )
            np.testing.assert_array_equal(
                conv2d1.bias.numpy(), conv2d2.bias.numpy()
            )

        with self.static_graph():

            # the input of Conv2DTranspose must be Variable.
            def test_Variable():
                images = np.ones([2, 3, 5, 5], dtype='float32')
                conv2d = paddle.nn.Conv2DTranspose(3, 3, [2, 2])
                conv2d_ret1 = conv2d(images)

            self.assertRaises(TypeError, test_Variable)

            # the input dtype of Conv2DTranspose must be float16 or float32 or float64
            # float16 only can be set on GPU place
            def test_type():
                images = layers.data(
                    name='pixel', shape=[3, 5, 5], dtype='int32'
                )
                conv2d = paddle.nn.Conv2DTranspose(3, 3, [2, 2])
                conv2d_ret2 = conv2d(images)

            self.assertRaises(TypeError, test_type)

    def test_bilinear_tensor_product(self):
        inp_np_x = np.array([[1, 2, 3]]).astype('float32')
        inp_np_y = np.array([[4, 5, 6]]).astype('float32')

        with self.static_graph():
            data_x = layers.data(
                name='x', shape=[1, 3], dtype="float32", append_batch_size=False
            )
            data_y = layers.data(
                name='y', shape=[1, 3], dtype="float32", append_batch_size=False
            )
            out = paddle.static.nn.common.bilinear_tensor_product(
                data_x,
                data_y,
                6,
                bias_attr=fluid.initializer.ConstantInitializer(value=1),
                act='sigmoid',
            )

            static_rlt = self.get_static_graph_result(
                feed={'x': inp_np_x, 'y': inp_np_y}, fetch_list=[out]
            )[0]

        with self.static_graph():
            data_x = layers.data(
                name='x', shape=[1, 3], dtype="float32", append_batch_size=False
            )
            data_y = layers.data(
                name='y', shape=[1, 3], dtype="float32", append_batch_size=False
            )
            btp = paddle.nn.Bilinear(
                3,
                3,
                6,
                bias_attr=fluid.initializer.ConstantInitializer(value=1),
            )
            out = btp(data_x, data_y)
            out = paddle.nn.functional.sigmoid(out)
            static_rlt2 = self.get_static_graph_result(
                feed={'x': inp_np_x, 'y': inp_np_y}, fetch_list=[out]
            )[0]
        with self.dynamic_graph():
            with _test_eager_guard():
                btp = paddle.nn.Bilinear(
                    3,
                    3,
                    6,
                    bias_attr=fluid.initializer.ConstantInitializer(value=1),
                )
                dy_eager_rlt = btp(
                    base.to_variable(inp_np_x), base.to_variable(inp_np_y)
                )
                dy_eager_rlt = paddle.nn.functional.sigmoid(dy_eager_rlt)
                dy_eager_rlt_value = dy_eager_rlt.numpy()

            btp = paddle.nn.Bilinear(
                3,
                3,
                6,
                bias_attr=fluid.initializer.ConstantInitializer(value=1),
            )
            dy_rlt = btp(base.to_variable(inp_np_x), base.to_variable(inp_np_y))
            dy_rlt = paddle.nn.functional.sigmoid(dy_rlt)
            dy_rlt_value = dy_rlt.numpy()

        with self.dynamic_graph():
            with _test_eager_guard():
                btp2 = paddle.nn.Bilinear(3, 3, 6)
                dy_eager_rlt2 = btp2(
                    base.to_variable(inp_np_x), base.to_variable(inp_np_y)
                )
                dy_eager_rlt2 = paddle.nn.functional.sigmoid(dy_eager_rlt2)
                dy_eager_rlt2_value = dy_eager_rlt2.numpy()

            btp2 = paddle.nn.Bilinear(3, 3, 6)
            dy_rlt2 = btp2(
                base.to_variable(inp_np_x), base.to_variable(inp_np_y)
            )
            dy_rlt2 = paddle.nn.functional.sigmoid(dy_rlt2)
            dy_rlt2_value = dy_rlt2.numpy()

        with self.static_graph():
            data_x2 = layers.data(
                name='x', shape=[1, 3], dtype="float32", append_batch_size=False
            )
            data_y2 = layers.data(
                name='y', shape=[1, 3], dtype="float32", append_batch_size=False
            )
            out2 = paddle.static.nn.common.bilinear_tensor_product(
                data_x2, data_y2, 6, act='sigmoid'
            )

            static_rlt3 = self.get_static_graph_result(
                feed={'x': inp_np_x, 'y': inp_np_y}, fetch_list=[out2]
            )[0]

        np.testing.assert_array_equal(dy_rlt2_value, static_rlt3)
        np.testing.assert_array_equal(dy_eager_rlt2_value, static_rlt3)
        np.testing.assert_array_equal(static_rlt2, static_rlt)
        np.testing.assert_array_equal(dy_rlt_value, static_rlt)
        np.testing.assert_array_equal(dy_eager_rlt_value, static_rlt)

        with self.dynamic_graph():
            with _test_eager_guard():
                custom_weight = np.random.randn(6, 3, 3).astype("float32")
                weight_attr = fluid.ParamAttr(
                    initializer=fluid.initializer.NumpyArrayInitializer(
                        custom_weight
                    )
                )
                btp1 = paddle.nn.Bilinear(3, 3, 6)
                btp2 = paddle.nn.Bilinear(3, 3, 6, weight_attr=weight_attr)
                dy_rlt1 = btp1(
                    base.to_variable(inp_np_x), base.to_variable(inp_np_y)
                )
                dy_rlt1 = paddle.nn.functional.sigmoid(dy_rlt1)
                dy_rlt2 = btp2(
                    base.to_variable(inp_np_x), base.to_variable(inp_np_y)
                )
                dy_rlt2 = paddle.nn.functional.sigmoid(dy_rlt2)
                self.assertFalse(
                    np.array_equal(dy_rlt1.numpy(), dy_rlt2.numpy())
                )
                btp2.weight.set_value(btp1.weight.numpy())
                btp2.bias.set_value(btp1.bias)
                dy_rlt1 = btp1(
                    base.to_variable(inp_np_x), base.to_variable(inp_np_y)
                )
                dy_rlt2 = btp2(
                    base.to_variable(inp_np_x), base.to_variable(inp_np_y)
                )
                np.testing.assert_array_equal(dy_rlt1.numpy(), dy_rlt2.numpy())

                btp2.weight = btp1.weight
                btp2.bias = btp1.bias
                np.testing.assert_array_equal(
                    btp1.weight.numpy(), btp2.weight.numpy()
                )
                np.testing.assert_array_equal(
                    btp1.bias.numpy(), btp2.bias.numpy()
                )

            custom_weight = np.random.randn(6, 3, 3).astype("float32")
            weight_attr = fluid.ParamAttr(
                initializer=fluid.initializer.NumpyArrayInitializer(
                    custom_weight
                )
            )
            btp1 = paddle.nn.Bilinear(3, 3, 6)
            btp2 = paddle.nn.Bilinear(3, 3, 6, weight_attr=weight_attr)
            dy_rlt1 = btp1(
                base.to_variable(inp_np_x), base.to_variable(inp_np_y)
            )
            dy_rlt1 = paddle.nn.functional.sigmoid(dy_rlt1)
            dy_rlt2 = btp2(
                base.to_variable(inp_np_x), base.to_variable(inp_np_y)
            )
            dy_rlt2 = paddle.nn.functional.sigmoid(dy_rlt2)
            self.assertFalse(np.array_equal(dy_rlt1.numpy(), dy_rlt2.numpy()))
            btp2.weight.set_value(btp1.weight.numpy())
            btp2.bias.set_value(btp1.bias)
            dy_rlt1 = btp1(
                base.to_variable(inp_np_x), base.to_variable(inp_np_y)
            )
            dy_rlt2 = btp2(
                base.to_variable(inp_np_x), base.to_variable(inp_np_y)
            )
            np.testing.assert_array_equal(dy_rlt1.numpy(), dy_rlt2.numpy())

            btp2.weight = btp1.weight
            btp2.bias = btp1.bias
            np.testing.assert_array_equal(
                btp1.weight.numpy(), btp2.weight.numpy()
            )
            np.testing.assert_array_equal(btp1.bias.numpy(), btp2.bias.numpy())

    def test_embeding(self):
        inp_word = np.array([[[1]]]).astype('int64')
        dict_size = 20
        with self.static_graph():
            data_t = layers.data(name='word', shape=[1], dtype='int64')
            emb = layers.embedding(
                input=data_t,
                size=[dict_size, 32],
                param_attr='emb.w',
                is_sparse=False,
            )
            static_rlt = self.get_static_graph_result(
                feed={'word': inp_word}, fetch_list=[emb]
            )[0]
        with self.static_graph():
            data_t = layers.data(name='word', shape=[1], dtype='int64')
            emb2 = paddle.nn.Embedding(
                dict_size, 32, weight_attr='emb.w', sparse=False
            )
            emb_rlt = emb2(data_t)
            static_rlt2 = self.get_static_graph_result(
                feed={'word': inp_word}, fetch_list=[emb_rlt]
            )[0]
        with self.dynamic_graph():
            with _test_eager_guard():
                emb2 = paddle.nn.Embedding(
                    dict_size,
                    32,
                    weight_attr='eager_emb.w',
                    sparse=False,
                )
                dy_eager_rlt = emb2(base.to_variable(inp_word))
                dy_eager_rlt_value = dy_eager_rlt.numpy()

            emb2 = paddle.nn.Embedding(
                dict_size, 32, weight_attr='emb.w', sparse=False
            )
            dy_rlt = emb2(base.to_variable(inp_word))
            dy_rlt_value = dy_rlt.numpy()

        self.assertTrue(np.allclose(static_rlt2, static_rlt))
        self.assertTrue(np.allclose(dy_rlt_value, static_rlt))
        self.assertTrue(np.allclose(dy_eager_rlt_value, static_rlt))

        with self.dynamic_graph():
            with _test_eager_guard():
                custom_weight = np.random.randn(dict_size, 32).astype("float32")
                weight_attr = fluid.ParamAttr(
                    initializer=fluid.initializer.NumpyArrayInitializer(
                        custom_weight
                    )
                )
                emb1 = paddle.nn.Embedding(dict_size, 32, sparse=False)
                emb2 = paddle.nn.Embedding(
                    dict_size,
                    32,
                    weight_attr=weight_attr,
                    sparse=False,
                )
                rep1 = emb1(base.to_variable(inp_word))
                rep2 = emb2(base.to_variable(inp_word))
                self.assertFalse(
                    np.array_equal(emb1.weight.numpy(), custom_weight)
                )
                np.testing.assert_array_equal(
                    emb2.weight.numpy(), custom_weight
                )
                self.assertFalse(np.array_equal(rep1.numpy(), rep2.numpy()))
                emb2.weight.set_value(emb1.weight.numpy())
                rep2 = emb2(base.to_variable(inp_word))
                np.testing.assert_array_equal(rep1.numpy(), rep2.numpy())

                emb2.weight = emb1.weight
                np.testing.assert_array_equal(
                    emb1.weight.numpy(), emb2.weight.numpy()
                )

            custom_weight = np.random.randn(dict_size, 32).astype("float32")
            weight_attr = fluid.ParamAttr(
                initializer=fluid.initializer.NumpyArrayInitializer(
                    custom_weight
                )
            )
            emb1 = paddle.nn.Embedding(dict_size, 32, sparse=False)
            emb2 = paddle.nn.Embedding(
                dict_size, 32, weight_attr=weight_attr, sparse=False
            )
            rep1 = emb1(base.to_variable(inp_word))
            rep2 = emb2(base.to_variable(inp_word))
            self.assertFalse(np.array_equal(emb1.weight.numpy(), custom_weight))
            np.testing.assert_array_equal(emb2.weight.numpy(), custom_weight)
            self.assertFalse(np.array_equal(rep1.numpy(), rep2.numpy()))
            emb2.weight.set_value(emb1.weight.numpy())
            rep2 = emb2(base.to_variable(inp_word))
            np.testing.assert_array_equal(rep1.numpy(), rep2.numpy())

            emb2.weight = emb1.weight
            np.testing.assert_array_equal(
                emb1.weight.numpy(), emb2.weight.numpy()
            )

    def test_one_hot(self):
        with self.dynamic_graph():
            with _test_eager_guard():
                label = fluid.dygraph.to_variable(
                    np.array([[1], [1], [3], [0]])
                )
                one_hot_label1 = fluid.layers.one_hot(input=label, depth=4)
                one_hot_label2 = fluid.layers.one_hot(
                    input=label, depth=fluid.dygraph.to_variable(np.array([4]))
                )
                np.testing.assert_array_equal(
                    one_hot_label1.numpy(), one_hot_label2.numpy()
                )

            label = fluid.dygraph.to_variable(np.array([[1], [1], [3], [0]]))
            one_hot_label1 = fluid.layers.one_hot(input=label, depth=4)
            one_hot_label2 = fluid.layers.one_hot(
                input=label, depth=fluid.dygraph.to_variable(np.array([4]))
            )
            np.testing.assert_array_equal(
                one_hot_label1.numpy(), one_hot_label2.numpy()
            )

    def test_split(self):
        with self.dynamic_graph():
            with _test_eager_guard():
                input = fluid.dygraph.to_variable(np.random.random((3, 8, 5)))
                x0, x1 = paddle.split(input, num_or_sections=2, axis=1)
                x00, x11 = paddle.split(
                    input,
                    num_or_sections=2,
                    axis=fluid.dygraph.to_variable(np.array([1])),
                )
                np.testing.assert_array_equal(x0.numpy(), x00.numpy())
                np.testing.assert_array_equal(x1.numpy(), x11.numpy())

            input = fluid.dygraph.to_variable(np.random.random((3, 8, 5)))
            x0, x1 = paddle.split(input, num_or_sections=2, axis=1)
            x00, x11 = paddle.split(
                input,
                num_or_sections=2,
                axis=fluid.dygraph.to_variable(np.array([1])),
            )
            np.testing.assert_array_equal(x0.numpy(), x00.numpy())
            np.testing.assert_array_equal(x1.numpy(), x11.numpy())

    def test_topk(self):
        with self.dynamic_graph():
            with _test_eager_guard():
                input = fluid.dygraph.to_variable(np.random.random((13, 11)))
                top5_values1, top5_indices1 = paddle.topk(input, k=5)
                top5_values2, top5_indices2 = paddle.topk(
                    input, k=fluid.dygraph.to_variable(np.array([5]))
                )
                np.testing.assert_array_equal(
                    top5_values1.numpy(), top5_values2.numpy()
                )
                np.testing.assert_array_equal(
                    top5_indices1.numpy(), top5_indices2.numpy()
                )

            input = fluid.dygraph.to_variable(np.random.random((13, 11)))
            top5_values1, top5_indices1 = paddle.topk(input, k=5)
            top5_values2, top5_indices2 = paddle.topk(
                input, k=fluid.dygraph.to_variable(np.array([5]))
            )
            np.testing.assert_array_equal(
                top5_values1.numpy(), top5_values2.numpy()
            )
            np.testing.assert_array_equal(
                top5_indices1.numpy(), top5_indices2.numpy()
            )

    def test_conv3d(self):
        with self.static_graph():
            images = layers.data(
                name='pixel', shape=[3, 6, 6, 6], dtype='float32'
            )
            ret = paddle.static.nn.conv3d(
                input=images, num_filters=3, filter_size=2
            )
            static_ret = self.get_static_graph_result(
                feed={'pixel': np.ones([2, 3, 6, 6, 6], dtype='float32')},
                fetch_list=[ret],
            )[0]

        with self.static_graph():
            images = layers.data(
                name='pixel', shape=[3, 6, 6, 6], dtype='float32'
            )
            conv3d = paddle.nn.Conv3D(
                in_channels=3, out_channels=3, kernel_size=2
            )
            ret = conv3d(images)
            static_ret2 = self.get_static_graph_result(
                feed={'pixel': np.ones([2, 3, 6, 6, 6], dtype='float32')},
                fetch_list=[ret],
            )[0]

        with self.dynamic_graph():
            with _test_eager_guard():
                images = np.ones([2, 3, 6, 6, 6], dtype='float32')
                conv3d = paddle.nn.Conv3D(
                    in_channels=3, out_channels=3, kernel_size=2
                )
                dy_eager_ret = conv3d(base.to_variable(images))
                dy_eager_rlt_value = dy_eager_ret.numpy()

            images = np.ones([2, 3, 6, 6, 6], dtype='float32')
            conv3d = paddle.nn.Conv3D(
                in_channels=3, out_channels=3, kernel_size=2
            )
            dy_ret = conv3d(base.to_variable(images))
            dy_rlt_value = dy_ret.numpy()

        np.testing.assert_allclose(static_ret, dy_rlt_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret, dy_eager_rlt_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret, static_ret2, rtol=1e-05)

        with self.dynamic_graph():
            with _test_eager_guard():
                images = np.ones([2, 3, 6, 6, 6], dtype='float32')
                custom_weight = np.random.randn(3, 3, 2, 2, 2).astype("float32")
                weight_attr = fluid.ParamAttr(
                    initializer=fluid.initializer.NumpyArrayInitializer(
                        custom_weight
                    )
                )
                conv3d1 = paddle.nn.Conv3D(
                    in_channels=3, out_channels=3, kernel_size=2
                )
                conv3d2 = paddle.nn.Conv3D(
                    in_channels=3,
                    out_channels=3,
                    kernel_size=2,
                    weight_attr=weight_attr,
                )
                dy_ret1 = conv3d1(base.to_variable(images))
                dy_ret2 = conv3d2(base.to_variable(images))
                self.assertFalse(
                    np.array_equal(dy_ret1.numpy(), dy_ret2.numpy())
                )

                conv3d1_weight_np = conv3d1.weight.numpy()
                conv3d1_bias = conv3d1.bias
                self.assertFalse(
                    np.array_equal(conv3d1_weight_np, conv3d2.weight.numpy())
                )
                conv3d2.weight.set_value(conv3d1_weight_np)
                np.testing.assert_array_equal(
                    conv3d1_weight_np, conv3d2.weight.numpy()
                )
                conv3d1.bias.set_value(conv3d1_bias)
                dy_ret1 = conv3d1(base.to_variable(images))
                dy_ret2 = conv3d2(base.to_variable(images))
                np.testing.assert_array_equal(dy_ret1.numpy(), dy_ret2.numpy())

                conv3d2.weight = conv3d1.weight
                conv3d2.bias = conv3d1.bias
                np.testing.assert_array_equal(
                    conv3d1.weight.numpy(), conv3d2.weight.numpy()
                )
                np.testing.assert_array_equal(
                    conv3d1.bias.numpy(), conv3d2.bias.numpy()
                )

            images = np.ones([2, 3, 6, 6, 6], dtype='float32')
            custom_weight = np.random.randn(3, 3, 2, 2, 2).astype("float32")
            weight_attr = fluid.ParamAttr(
                initializer=fluid.initializer.NumpyArrayInitializer(
                    custom_weight
                )
            )
            conv3d1 = paddle.nn.Conv3D(
                in_channels=3, out_channels=3, kernel_size=2
            )
            conv3d2 = paddle.nn.Conv3D(
                in_channels=3,
                out_channels=3,
                kernel_size=2,
                weight_attr=weight_attr,
            )
            dy_ret1 = conv3d1(base.to_variable(images))
            dy_ret2 = conv3d2(base.to_variable(images))
            self.assertFalse(np.array_equal(dy_ret1.numpy(), dy_ret2.numpy()))

            conv3d1_weight_np = conv3d1.weight.numpy()
            conv3d1_bias = conv3d1.bias
            self.assertFalse(
                np.array_equal(conv3d1_weight_np, conv3d2.weight.numpy())
            )
            conv3d2.weight.set_value(conv3d1_weight_np)
            np.testing.assert_array_equal(
                conv3d1_weight_np, conv3d2.weight.numpy()
            )
            conv3d1.bias.set_value(conv3d1_bias)
            dy_ret1 = conv3d1(base.to_variable(images))
            dy_ret2 = conv3d2(base.to_variable(images))
            np.testing.assert_array_equal(dy_ret1.numpy(), dy_ret2.numpy())

            conv3d2.weight = conv3d1.weight
            conv3d2.bias = conv3d1.bias
            np.testing.assert_array_equal(
                conv3d1.weight.numpy(), conv3d2.weight.numpy()
            )
            np.testing.assert_array_equal(
                conv3d1.bias.numpy(), conv3d2.bias.numpy()
            )

    def func_group_norm(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()

        shape = (2, 4, 3, 3)

        input = np.random.random(shape).astype('float32')

        with self.static_graph():
            X = fluid.layers.data(
                name='X',
                shape=shape,
                dtype='float32',
                lod_level=1,
                append_batch_size=False,
            )
            ret = paddle.static.nn.group_norm(
                input=X,
                groups=2,
                param_attr=fluid.initializer.Uniform(low=-0.5, high=0.5),
                bias_attr=fluid.initializer.ConstantInitializer(value=1),
            )
            static_ret = self.get_static_graph_result(
                feed={
                    'X': fluid.create_lod_tensor(
                        data=input, recursive_seq_lens=[[1, 1]], place=place
                    )
                },
                fetch_list=[ret],
                with_lod=True,
            )[0]

        with self.static_graph():
            X = fluid.layers.data(
                name='X',
                shape=shape,
                dtype='float32',
                lod_level=1,
                append_batch_size=False,
            )
            groupNorm = paddle.nn.GroupNorm(
                num_channels=shape[1],
                num_groups=2,
                weight_attr=fluid.initializer.Uniform(low=-0.5, high=0.5),
                bias_attr=fluid.initializer.ConstantInitializer(value=1),
            )
            ret = groupNorm(X)
            static_ret2 = self.get_static_graph_result(
                feed={
                    'X': fluid.create_lod_tensor(
                        data=input, recursive_seq_lens=[[1, 1]], place=place
                    )
                },
                fetch_list=[ret],
                with_lod=True,
            )[0]

        with self.dynamic_graph():
            groupNorm = paddle.nn.GroupNorm(
                num_channels=shape[1],
                num_groups=2,
                weight_attr=fluid.initializer.Uniform(low=-0.5, high=0.5),
                bias_attr=fluid.initializer.ConstantInitializer(value=1),
            )
            dy_ret = groupNorm(base.to_variable(input))
            dy_rlt_value = dy_ret.numpy()

        np.testing.assert_allclose(static_ret, dy_rlt_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret, static_ret2, rtol=1e-05)

    def test_group_norm(self):
        with _test_eager_guard():
            self.func_group_norm()
        self.func_group_norm()

    def test_instance_norm(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()

        shape = (2, 4, 3, 3)

        input = np.random.random(shape).astype('float32')

        with self.static_graph():
            X = fluid.layers.data(
                name='X', shape=shape, dtype='float32', append_batch_size=False
            )
            ret = paddle.static.nn.instance_norm(input=X)
            static_ret = self.get_static_graph_result(
                feed={'X': input}, fetch_list=[ret]
            )[0]

        with self.static_graph():
            X = fluid.layers.data(
                name='X', shape=shape, dtype='float32', append_batch_size=False
            )
            instanceNorm = paddle.nn.InstanceNorm2D(num_features=shape[1])
            ret = instanceNorm(X)
            static_ret2 = self.get_static_graph_result(
                feed={'X': input}, fetch_list=[ret]
            )[0]

        with self.dynamic_graph():
            with _test_eager_guard():
                instanceNorm = paddle.nn.InstanceNorm2D(num_features=shape[1])
                dy_eager_ret = instanceNorm(base.to_variable(input))
                dy_eager_rlt_value = dy_eager_ret.numpy()

            instanceNorm = paddle.nn.InstanceNorm2D(num_features=shape[1])
            dy_ret = instanceNorm(base.to_variable(input))
            dy_rlt_value = dy_ret.numpy()

        with self.dynamic_graph():
            with _test_eager_guard():
                instanceNorm = paddle.nn.InstanceNorm2D(num_features=shape[1])
                dy_eager_ret = instanceNorm(base.to_variable(input))
                dy_eager_rlt_value2 = dy_eager_ret.numpy()

            instanceNorm = paddle.nn.InstanceNorm2D(num_features=shape[1])
            dy_ret = instanceNorm(base.to_variable(input))
            dy_rlt_value2 = dy_ret.numpy()

        np.testing.assert_allclose(static_ret, dy_rlt_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret, dy_rlt_value2, rtol=1e-05)
        np.testing.assert_allclose(static_ret, dy_eager_rlt_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret, dy_eager_rlt_value2, rtol=1e-05)
        np.testing.assert_allclose(static_ret, static_ret2, rtol=1e-05)

        with self.static_graph():
            # the input of InstanceNorm must be Variable.
            def test_Variable():
                instanceNorm = paddle.nn.InstanceNorm2D(num_features=shape[1])
                ret1 = instanceNorm(input)

            self.assertRaises(TypeError, test_Variable)

            # the input dtype of InstanceNorm must be float32 or float64
            def test_type():
                input = np.random.random(shape).astype('int32')
                instanceNorm = paddle.nn.InstanceNorm2D(num_features=shape[1])
                ret2 = instanceNorm(input)

            self.assertRaises(TypeError, test_type)

    def test_spectral_norm(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()

        shape = (2, 4, 3, 3)

        input = np.random.random(shape).astype('float32')

        with self.static_graph():
            Weight = fluid.layers.data(
                name='Weight',
                shape=shape,
                dtype='float32',
                lod_level=1,
                append_batch_size=False,
            )
            ret = layers.spectral_norm(weight=Weight, dim=1, power_iters=2)
            static_ret = self.get_static_graph_result(
                feed={
                    'Weight': fluid.create_lod_tensor(
                        data=input, recursive_seq_lens=[[1, 1]], place=place
                    ),
                },
                fetch_list=[ret],
                with_lod=True,
            )[0]

        with self.static_graph():
            Weight = fluid.layers.data(
                name='Weight',
                shape=shape,
                dtype='float32',
                lod_level=1,
                append_batch_size=False,
            )
            spectralNorm = paddle.nn.SpectralNorm(shape, axis=1, power_iters=2)
            ret = spectralNorm(Weight)
            static_ret2 = self.get_static_graph_result(
                feed={
                    'Weight': fluid.create_lod_tensor(
                        data=input, recursive_seq_lens=[[1, 1]], place=place
                    )
                },
                fetch_list=[ret],
                with_lod=True,
            )[0]

        with self.dynamic_graph():
            with _test_eager_guard():
                spectralNorm = paddle.nn.SpectralNorm(
                    shape, axis=1, power_iters=2
                )
                dy_eager_ret = spectralNorm(base.to_variable(input))
                dy_eager_rlt_value = dy_eager_ret.numpy()

            spectralNorm = paddle.nn.SpectralNorm(shape, axis=1, power_iters=2)
            dy_ret = spectralNorm(base.to_variable(input))
            dy_rlt_value = dy_ret.numpy()

        np.testing.assert_allclose(static_ret, dy_rlt_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret, dy_eager_rlt_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret, static_ret2, rtol=1e-05)

    def test_conv3d_transpose(self):
        input_array = (
            np.arange(0, 48).reshape([2, 3, 2, 2, 2]).astype('float32')
        )

        with self.static_graph():
            img = layers.data(name='pixel', shape=[3, 2, 2, 2], dtype='float32')
            out = paddle.static.nn.conv3d_transpose(
                input=img, num_filters=12, filter_size=12, use_cudnn=True
            )
            static_rlt = self.get_static_graph_result(
                feed={'pixel': input_array}, fetch_list=[out]
            )[0]
        with self.static_graph():
            img = layers.data(name='pixel', shape=[3, 2, 2, 2], dtype='float32')
            conv3d_transpose = paddle.nn.Conv3DTranspose(
                in_channels=3, out_channels=12, kernel_size=12
            )
            out = conv3d_transpose(img)
            static_rlt2 = self.get_static_graph_result(
                feed={'pixel': input_array}, fetch_list=[out]
            )[0]
        with self.dynamic_graph():
            with _test_eager_guard():
                conv3d_transpose = paddle.nn.Conv3DTranspose(
                    in_channels=3,
                    out_channels=12,
                    kernel_size=12,
                )
                dy_eager_rlt = conv3d_transpose(base.to_variable(input_array))
                dy_eager_rlt_value = dy_eager_rlt.numpy()

            conv3d_transpose = paddle.nn.Conv3DTranspose(
                in_channels=3, out_channels=12, kernel_size=12
            )
            dy_rlt = conv3d_transpose(base.to_variable(input_array))
            dy_rlt_value = dy_rlt.numpy()
        np.testing.assert_allclose(static_rlt2, static_rlt, rtol=1e-05)
        np.testing.assert_allclose(dy_rlt_value, static_rlt, rtol=1e-05)
        np.testing.assert_allclose(dy_eager_rlt_value, static_rlt, rtol=1e-05)

        with self.dynamic_graph():
            with _test_eager_guard():
                images = np.ones([2, 3, 6, 6, 6], dtype='float32')
                custom_weight = np.random.randn(3, 3, 2, 2, 2).astype("float32")
                weight_attr = fluid.ParamAttr(
                    initializer=fluid.initializer.NumpyArrayInitializer(
                        custom_weight
                    )
                )
                conv3d1 = paddle.nn.Conv3DTranspose(
                    in_channels=3,
                    out_channels=3,
                    kernel_size=2,
                    bias_attr='eager_conv3d1_b',
                )
                conv3d2 = paddle.nn.Conv3DTranspose(
                    in_channels=3,
                    out_channels=3,
                    kernel_size=2,
                    weight_attr=weight_attr,
                    bias_attr='eager_conv3d2_b',
                )
                dy_ret1 = conv3d1(base.to_variable(images))
                dy_ret2 = conv3d2(base.to_variable(images))
                self.assertFalse(
                    np.array_equal(dy_ret1.numpy(), dy_ret2.numpy())
                )

                conv3d1_weight_np = conv3d1.weight.numpy()
                conv3d1_bias = conv3d1.bias
                self.assertFalse(
                    np.array_equal(conv3d1_weight_np, conv3d2.weight.numpy())
                )
                conv3d2.weight.set_value(conv3d1_weight_np)
                np.testing.assert_array_equal(
                    conv3d1_weight_np, conv3d2.weight.numpy()
                )
                conv3d1.bias.set_value(conv3d1_bias)
                dy_ret1 = conv3d1(base.to_variable(images))
                dy_ret2 = conv3d2(base.to_variable(images))
                np.testing.assert_array_equal(dy_ret1.numpy(), dy_ret2.numpy())

                conv3d2.weight = conv3d1.weight
                conv3d2.bias = conv3d1.bias
                np.testing.assert_array_equal(
                    conv3d1.weight.numpy(), conv3d2.weight.numpy()
                )
                np.testing.assert_array_equal(
                    conv3d1.bias.numpy(), conv3d2.bias.numpy()
                )

            images = np.ones([2, 3, 6, 6, 6], dtype='float32')
            custom_weight = np.random.randn(3, 3, 2, 2, 2).astype("float32")
            weight_attr = fluid.ParamAttr(
                initializer=fluid.initializer.NumpyArrayInitializer(
                    custom_weight
                )
            )
            conv3d1 = paddle.nn.Conv3DTranspose(
                in_channels=3,
                out_channels=3,
                kernel_size=2,
                bias_attr='conv3d1_b',
            )
            conv3d2 = paddle.nn.Conv3DTranspose(
                in_channels=3,
                out_channels=3,
                kernel_size=2,
                weight_attr=weight_attr,
                bias_attr='conv3d2_b',
            )
            dy_ret1 = conv3d1(base.to_variable(images))
            dy_ret2 = conv3d2(base.to_variable(images))
            self.assertFalse(np.array_equal(dy_ret1.numpy(), dy_ret2.numpy()))

            conv3d1_weight_np = conv3d1.weight.numpy()
            conv3d1_bias = conv3d1.bias
            self.assertFalse(
                np.array_equal(conv3d1_weight_np, conv3d2.weight.numpy())
            )
            conv3d2.weight.set_value(conv3d1_weight_np)
            np.testing.assert_array_equal(
                conv3d1_weight_np, conv3d2.weight.numpy()
            )
            conv3d1.bias.set_value(conv3d1_bias)
            dy_ret1 = conv3d1(base.to_variable(images))
            dy_ret2 = conv3d2(base.to_variable(images))
            np.testing.assert_array_equal(dy_ret1.numpy(), dy_ret2.numpy())

            conv3d2.weight = conv3d1.weight
            conv3d2.bias = conv3d1.bias
            np.testing.assert_array_equal(
                conv3d1.weight.numpy(), conv3d2.weight.numpy()
            )
            np.testing.assert_array_equal(
                conv3d1.bias.numpy(), conv3d2.bias.numpy()
            )

    def func_while_loop(self):
        with self.static_graph():
            i = layers.fill_constant(shape=[1], dtype='int64', value=0)
            ten = layers.fill_constant(shape=[1], dtype='int64', value=10)

            def cond(i):
                return paddle.less_than(i, ten)

            def body(i):
                return i + 1

            out = paddle.static.nn.while_loop(cond, body, [i])
            static_ret = self.get_static_graph_result(feed={}, fetch_list=out)

        with self.dynamic_graph():
            i = layers.fill_constant(shape=[1], dtype='int64', value=0)
            ten = layers.fill_constant(shape=[1], dtype='int64', value=10)

            def cond1(i):
                return paddle.less_than(i, ten)

            def body1(i):
                return i + 1

            dy_ret = paddle.static.nn.while_loop(cond1, body1, [i])
            with self.assertRaises(ValueError):
                j = layers.fill_constant(shape=[1], dtype='int64', value=0)

                def body2(i):
                    return i + 1, i + 2

                paddle.static.nn.while_loop(cond1, body2, [j])

        np.testing.assert_array_equal(static_ret[0], dy_ret[0].numpy())

    def test_while_loop(self):
        with _test_eager_guard():
            self.func_while_loop()
        self.func_while_loop()

    def test_compare(self):
        value_a = np.arange(3)
        value_b = np.arange(3)
        # less than
        with self.static_graph():
            a = layers.data(name='a', shape=[1], dtype='int64')
            b = layers.data(name='b', shape=[1], dtype='int64')
            cond = paddle.less_than(x=a, y=b)
            static_ret = self.get_static_graph_result(
                feed={"a": value_a, "b": value_b}, fetch_list=[cond]
            )[0]
        with self.dynamic_graph():
            with _test_eager_guard():
                da = base.to_variable(value_a)
                db = base.to_variable(value_b)
                dcond = paddle.less_than(x=da, y=db)

                for i in range(len(static_ret)):
                    self.assertTrue(dcond.numpy()[i] == static_ret[i])

            da = base.to_variable(value_a)
            db = base.to_variable(value_b)
            dcond = paddle.less_than(x=da, y=db)

            for i in range(len(static_ret)):
                self.assertTrue(dcond.numpy()[i] == static_ret[i])

        # less equal
        with self.static_graph():
            a1 = layers.data(name='a1', shape=[1], dtype='int64')
            b1 = layers.data(name='b1', shape=[1], dtype='int64')
            cond1 = paddle.less_equal(x=a1, y=b1)
            static_ret1 = self.get_static_graph_result(
                feed={"a1": value_a, "b1": value_b}, fetch_list=[cond1]
            )[0]
        with self.dynamic_graph():
            with _test_eager_guard():
                da1 = base.to_variable(value_a)
                db1 = base.to_variable(value_b)
                dcond1 = paddle.less_equal(x=da1, y=db1)

                for i in range(len(static_ret1)):
                    self.assertTrue(dcond1.numpy()[i] == static_ret1[i])

            da1 = base.to_variable(value_a)
            db1 = base.to_variable(value_b)
            dcond1 = paddle.less_equal(x=da1, y=db1)

            for i in range(len(static_ret1)):
                self.assertTrue(dcond1.numpy()[i] == static_ret1[i])

        # greater than
        with self.static_graph():
            a2 = layers.data(name='a2', shape=[1], dtype='int64')
            b2 = layers.data(name='b2', shape=[1], dtype='int64')
            cond2 = paddle.greater_than(x=a2, y=b2)
            static_ret2 = self.get_static_graph_result(
                feed={"a2": value_a, "b2": value_b}, fetch_list=[cond2]
            )[0]
        with self.dynamic_graph():
            with _test_eager_guard():
                da2 = base.to_variable(value_a)
                db2 = base.to_variable(value_b)
                dcond2 = paddle.greater_than(x=da2, y=db2)

                for i in range(len(static_ret2)):
                    self.assertTrue(dcond2.numpy()[i] == static_ret2[i])

            da2 = base.to_variable(value_a)
            db2 = base.to_variable(value_b)
            dcond2 = paddle.greater_than(x=da2, y=db2)

            for i in range(len(static_ret2)):
                self.assertTrue(dcond2.numpy()[i] == static_ret2[i])

        # greater equal
        with self.static_graph():
            a3 = layers.data(name='a3', shape=[1], dtype='int64')
            b3 = layers.data(name='b3', shape=[1], dtype='int64')
            cond3 = paddle.greater_equal(x=a3, y=b3)
            static_ret3 = self.get_static_graph_result(
                feed={"a3": value_a, "b3": value_b}, fetch_list=[cond3]
            )[0]
        with self.dynamic_graph():
            with _test_eager_guard():
                da3 = base.to_variable(value_a)
                db3 = base.to_variable(value_b)
                dcond3 = paddle.greater_equal(x=da3, y=db3)

                for i in range(len(static_ret3)):
                    self.assertTrue(dcond3.numpy()[i] == static_ret3[i])

            da3 = base.to_variable(value_a)
            db3 = base.to_variable(value_b)
            dcond3 = paddle.greater_equal(x=da3, y=db3)

            for i in range(len(static_ret3)):
                self.assertTrue(dcond3.numpy()[i] == static_ret3[i])

        # equal
        with self.static_graph():
            a4 = layers.data(name='a4', shape=[1], dtype='int64')
            b4 = layers.data(name='b4', shape=[1], dtype='int64')
            cond4 = paddle.equal(x=a4, y=b4)
            static_ret4 = self.get_static_graph_result(
                feed={"a4": value_a, "b4": value_b}, fetch_list=[cond4]
            )[0]
        with self.dynamic_graph():
            with _test_eager_guard():
                da4 = base.to_variable(value_a)
                db4 = base.to_variable(value_b)
                dcond4 = paddle.equal(x=da4, y=db4)

                for i in range(len(static_ret4)):
                    self.assertTrue(dcond4.numpy()[i] == static_ret4[i])

            da4 = base.to_variable(value_a)
            db4 = base.to_variable(value_b)
            dcond4 = paddle.equal(x=da4, y=db4)

            for i in range(len(static_ret4)):
                self.assertTrue(dcond4.numpy()[i] == static_ret4[i])

        # not equal
        with self.static_graph():
            a5 = layers.data(name='a5', shape=[1], dtype='int64')
            b5 = layers.data(name='b5', shape=[1], dtype='int64')
            cond5 = paddle.equal(x=a5, y=b5)
            static_ret5 = self.get_static_graph_result(
                feed={"a5": value_a, "b5": value_b}, fetch_list=[cond5]
            )[0]
        with self.dynamic_graph():
            with _test_eager_guard():
                da5 = base.to_variable(value_a)
                db5 = base.to_variable(value_b)
                dcond5 = paddle.equal(x=da5, y=db5)

                for i in range(len(static_ret5)):
                    self.assertTrue(dcond5.numpy()[i] == static_ret5[i])

            da5 = base.to_variable(value_a)
            db5 = base.to_variable(value_b)
            dcond5 = paddle.equal(x=da5, y=db5)

            for i in range(len(static_ret5)):
                self.assertTrue(dcond5.numpy()[i] == static_ret5[i])

    def test_cond(self):
        def less_than_branch(a, b):
            return paddle.add(a, b)

        def greater_equal_branch(a, b):
            return paddle.subtract(a, b)

        with self.static_graph():
            a = fluid.layers.fill_constant(
                shape=[1], dtype='float32', value=0.1
            )
            b = fluid.layers.fill_constant(
                shape=[1], dtype='float32', value=0.23
            )
            out = paddle.static.nn.cond(
                a >= b,
                lambda: greater_equal_branch(a, b),
                lambda: less_than_branch(a, b),
            )
            place = (
                fluid.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else fluid.CPUPlace()
            )
            exe = fluid.Executor(place)
            ret = exe.run(fetch_list=[out])
            static_res = ret[0]

        with self.dynamic_graph():
            with _test_eager_guard():
                a = fluid.dygraph.to_variable(np.array([0.1]).astype('float32'))
                b = fluid.dygraph.to_variable(
                    np.array([0.23]).astype('float32')
                )
                out = paddle.static.nn.cond(
                    a < b,
                    lambda: less_than_branch(a, b),
                    lambda: greater_equal_branch(a, b),
                )
                out2 = paddle.static.nn.cond(
                    a >= b,
                    lambda: greater_equal_branch(a, b),
                    lambda: less_than_branch(a, b),
                )
                eager_dynamic_res = out.numpy()
                eager_dynamic_res2 = out2.numpy()
                np.testing.assert_array_equal(
                    eager_dynamic_res, eager_dynamic_res2
                )
                with self.assertRaises(TypeError):
                    paddle.static.nn.cond(a < b, 'str', 'str')
                with self.assertRaises(TypeError):
                    paddle.static.nn.cond(a >= b, 'str', 'str')

            a = fluid.dygraph.to_variable(np.array([0.1]).astype('float32'))
            b = fluid.dygraph.to_variable(np.array([0.23]).astype('float32'))
            out = paddle.static.nn.cond(
                a < b,
                lambda: less_than_branch(a, b),
                lambda: greater_equal_branch(a, b),
            )
            out2 = paddle.static.nn.cond(
                a >= b,
                lambda: greater_equal_branch(a, b),
                lambda: less_than_branch(a, b),
            )
            dynamic_res = out.numpy()
            dynamic_res2 = out2.numpy()
            np.testing.assert_array_equal(dynamic_res, dynamic_res2)
            with self.assertRaises(TypeError):
                paddle.static.nn.cond(a < b, 'str', 'str')
            with self.assertRaises(TypeError):
                paddle.static.nn.cond(a >= b, 'str', 'str')

        np.testing.assert_array_equal(static_res, dynamic_res)
        np.testing.assert_array_equal(static_res, eager_dynamic_res)

    def test_case(self):
        def fn_1():
            return layers.fill_constant(shape=[1, 2], dtype='float32', value=1)

        def fn_2():
            return layers.fill_constant(shape=[2, 2], dtype='int32', value=2)

        def fn_3():
            return layers.fill_constant(shape=[3], dtype='int32', value=3)

        with self.static_graph():
            x = layers.fill_constant(shape=[1], dtype='float32', value=0.3)
            y = layers.fill_constant(shape=[1], dtype='float32', value=0.1)
            z = layers.fill_constant(shape=[1], dtype='float32', value=0.2)

            pred_1 = paddle.less_than(z, x)  # true: 0.2 < 0.3
            pred_2 = paddle.less_than(x, y)  # false: 0.3 < 0.1
            pred_3 = paddle.equal(x, y)  # false: 0.3 == 0.1

            out_1 = paddle.static.nn.case(
                pred_fn_pairs=[(pred_1, fn_1), (pred_2, fn_2)], default=fn_3
            )
            out_2 = paddle.static.nn.case(
                pred_fn_pairs=[(pred_2, fn_2), (pred_3, fn_3)]
            )

            place = (
                fluid.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else fluid.CPUPlace()
            )
            exe = fluid.Executor(place)
            static_res1, static_res2 = exe.run(fetch_list=[out_1, out_2])

        with self.dynamic_graph():
            with _test_eager_guard():
                x = layers.fill_constant(shape=[1], dtype='float32', value=0.3)
                y = layers.fill_constant(shape=[1], dtype='float32', value=0.1)
                z = layers.fill_constant(shape=[1], dtype='float32', value=0.2)

                pred_1 = paddle.less_than(z, x)  # true: 0.2 < 0.3
                pred_2 = paddle.less_than(x, y)  # false: 0.3 < 0.1
                pred_3 = paddle.equal(x, y)  # false: 0.3 == 0.1

                out_1 = paddle.static.nn.case(
                    pred_fn_pairs=[(pred_1, fn_1), (pred_2, fn_2)], default=fn_3
                )
                out_2 = paddle.static.nn.case(
                    pred_fn_pairs=[(pred_2, fn_2), (pred_3, fn_3)]
                )
                eager_dynamic_res1 = out_1.numpy()
                eager_dynamic_res2 = out_2.numpy()

            x = layers.fill_constant(shape=[1], dtype='float32', value=0.3)
            y = layers.fill_constant(shape=[1], dtype='float32', value=0.1)
            z = layers.fill_constant(shape=[1], dtype='float32', value=0.2)

            pred_1 = paddle.less_than(z, x)  # true: 0.2 < 0.3
            pred_2 = paddle.less_than(x, y)  # false: 0.3 < 0.1
            pred_3 = paddle.equal(x, y)  # false: 0.3 == 0.1

            out_1 = paddle.static.nn.case(
                pred_fn_pairs=[(pred_1, fn_1), (pred_2, fn_2)], default=fn_3
            )
            out_2 = paddle.static.nn.case(
                pred_fn_pairs=[(pred_2, fn_2), (pred_3, fn_3)]
            )
            dynamic_res1 = out_1.numpy()
            dynamic_res2 = out_2.numpy()

        np.testing.assert_array_equal(static_res1, dynamic_res1)
        np.testing.assert_array_equal(static_res2, dynamic_res2)
        np.testing.assert_array_equal(static_res1, eager_dynamic_res1)
        np.testing.assert_array_equal(static_res2, eager_dynamic_res2)

    def test_switch_case(self):
        def fn_1():
            return layers.fill_constant(shape=[1, 2], dtype='float32', value=1)

        def fn_2():
            return layers.fill_constant(shape=[2, 2], dtype='int32', value=2)

        def fn_3():
            return layers.fill_constant(shape=[3], dtype='int32', value=3)

        with self.static_graph():
            index_1 = layers.fill_constant(shape=[1], dtype='int32', value=1)
            index_2 = layers.fill_constant(shape=[1], dtype='int32', value=2)

            out_1 = paddle.static.nn.switch_case(
                branch_index=index_1,
                branch_fns={1: fn_1, 2: fn_2},
                default=fn_3,
            )
            out_2 = paddle.static.nn.switch_case(
                branch_index=index_2,
                branch_fns=[(1, fn_1), (2, fn_2)],
                default=fn_3,
            )
            out_3 = paddle.static.nn.switch_case(
                branch_index=index_2,
                branch_fns=[(0, fn_1), (4, fn_2), (7, fn_3)],
            )

            place = (
                fluid.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else fluid.CPUPlace()
            )
            exe = fluid.Executor(place)
            static_res1, static_res2, static_res3 = exe.run(
                fetch_list=[out_1, out_2, out_3]
            )

        with self.dynamic_graph():
            with _test_eager_guard():
                index_1 = layers.fill_constant(
                    shape=[1], dtype='int32', value=1
                )
                index_2 = layers.fill_constant(
                    shape=[1], dtype='int32', value=2
                )

                out_1 = paddle.static.nn.switch_case(
                    branch_index=index_1,
                    branch_fns={1: fn_1, 2: fn_2},
                    default=fn_3,
                )
                out_2 = paddle.static.nn.switch_case(
                    branch_index=index_2,
                    branch_fns=[(1, fn_1), (2, fn_2)],
                    default=fn_3,
                )
                out_3 = paddle.static.nn.switch_case(
                    branch_index=index_2,
                    branch_fns=[(0, fn_1), (4, fn_2), (7, fn_3)],
                )

                eager_dynamic_res1 = out_1.numpy()
                eager_dynamic_res2 = out_2.numpy()
                eager_dynamic_res3 = out_3.numpy()

            index_1 = layers.fill_constant(shape=[1], dtype='int32', value=1)
            index_2 = layers.fill_constant(shape=[1], dtype='int32', value=2)

            out_1 = paddle.static.nn.switch_case(
                branch_index=index_1,
                branch_fns={1: fn_1, 2: fn_2},
                default=fn_3,
            )
            out_2 = paddle.static.nn.switch_case(
                branch_index=index_2,
                branch_fns=[(1, fn_1), (2, fn_2)],
                default=fn_3,
            )
            out_3 = paddle.static.nn.switch_case(
                branch_index=index_2,
                branch_fns=[(0, fn_1), (4, fn_2), (7, fn_3)],
            )

            dynamic_res1 = out_1.numpy()
            dynamic_res2 = out_2.numpy()
            dynamic_res3 = out_3.numpy()

        np.testing.assert_array_equal(static_res1, dynamic_res1)
        np.testing.assert_array_equal(static_res2, dynamic_res2)
        np.testing.assert_array_equal(static_res3, dynamic_res3)
        np.testing.assert_array_equal(static_res1, eager_dynamic_res1)
        np.testing.assert_array_equal(static_res2, eager_dynamic_res2)
        np.testing.assert_array_equal(static_res3, eager_dynamic_res3)

    def test_crop_tensor(self):
        with self.static_graph():
            x = fluid.layers.data(name="x1", shape=[6, 5, 8])

            dim1 = fluid.layers.data(
                name="dim1", shape=[1], append_batch_size=False
            )
            dim2 = fluid.layers.data(
                name="dim2", shape=[1], append_batch_size=False
            )
            crop_shape1 = (1, 2, 4, 4)
            crop_shape2 = fluid.layers.data(
                name="crop_shape", shape=[4], append_batch_size=False
            )
            crop_shape3 = [-1, dim1, dim2, 4]
            crop_offsets1 = [0, 0, 1, 0]
            crop_offsets2 = fluid.layers.data(
                name="crop_offset", shape=[4], append_batch_size=False
            )
            crop_offsets3 = [0, dim1, dim2, 0]

            out1 = paddle.crop(x, shape=crop_shape1, offsets=crop_offsets1)
            out2 = paddle.crop(x, shape=crop_shape2, offsets=crop_offsets2)
            out3 = paddle.crop(x, shape=crop_shape3, offsets=crop_offsets3)

            self.assertIsNotNone(out1)
            self.assertIsNotNone(out2)
            self.assertIsNotNone(out3)

    def test_shard_index(self):
        with self.static_graph():
            x = fluid.layers.data(name="label", shape=[4, 1], dtype='int64')
            shard_label = paddle.shard_index(
                input=x, index_num=20, nshards=2, shard_id=0
            )

        self.assertIsNotNone(shard_label)

    def test_accuracy(self):
        x = np.random.rand(3, 32, 32).astype("float32")
        y = np.array([[1], [0], [1]])
        with self.static_graph():
            data = fluid.data(name="input", shape=[-1, 32, 32], dtype="float32")
            label = fluid.data(name="label", shape=[-1, 1], dtype="int")
            fc_out = fluid.layers.fc(input=data, size=10)
            predict = paddle.nn.functional.softmax(fc_out)
            result = paddle.static.accuracy(input=predict, label=label, k=5)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)

            exe.run(fluid.default_startup_program())
            # x = np.random.rand(3, 32, 32).astype("float32")
            # y = np.array([[1], [0], [1]])
            static_out = exe.run(
                feed={"input": x, "label": y}, fetch_list=result[0]
            )

        with self.dynamic_graph(force_to_use_cpu=True):
            data = base.to_variable(x)
            label = base.to_variable(y)
            fc_out = fluid.layers.fc(data, size=10)
            predict = paddle.nn.functional.softmax(fc_out)
            dynamic_out = paddle.static.accuracy(
                input=predict, label=label, k=5
            )

        np.testing.assert_array_equal(static_out[0], dynamic_out.numpy())


class TestBook(LayerTest):
    def setUp(self):
        self.only_static_set = set({"make_word_embedding"})
        self.not_compare_static_dygraph_set = set(
            {
                "make_gaussian_random",
                "make_kldiv_loss",
                "make_uniform_random_batch_size_like",
            }
        )
        self.all_close_compare = set({"make_spectral_norm"})

    def func_all_layers(self):
        attrs = (getattr(self, name) for name in dir(self))
        methods = filter(inspect.ismethod, attrs)
        for method in methods:
            if not method.__name__.startswith('make_'):
                continue
            self._low_data_bound = 0
            self._high_data_bound = 2
            self._batch_size = 2
            self._feed_dict = {}
            self._force_to_use_cpu = False
            with self.static_graph():
                static_var = method()
                if isinstance(static_var, tuple):
                    static_var = static_var[0]

                if static_var is not None:
                    fetch_list = [static_var.name]
                    static_result = self.get_static_graph_result(
                        feed=self._feed_dict,
                        fetch_list=fetch_list,
                        force_to_use_cpu=self._force_to_use_cpu,
                    )

                else:
                    continue
            if method.__name__ in self.only_static_set:
                continue

            with self.dynamic_graph(self._force_to_use_cpu):
                dy_result = method()
                if isinstance(dy_result, tuple):
                    dy_result = dy_result[0]
                dy_result_value = dy_result.numpy()

            if method.__name__ in self.all_close_compare:
                np.testing.assert_allclose(
                    static_result[0],
                    dy_result_value,
                    rtol=1e-05,
                    atol=0,
                    err_msg='Result of function [{}] compare failed'.format(
                        method.__name__
                    ),
                )
                continue

            if method.__name__ not in self.not_compare_static_dygraph_set:
                np.testing.assert_array_equal(
                    static_result[0],
                    dy_result_value,
                    err_msg='Result of function [{}] not equal'.format(
                        method.__name__
                    ),
                )

    def test_all_layers(self):
        with _test_eager_guard():
            self.func_all_layers()
        self.func_all_layers()

    def _get_np_data(self, shape, dtype, append_batch_size=True):
        np.random.seed(self.seed)
        if append_batch_size:
            shape = [self._batch_size] + shape
        if dtype == 'float32':
            return np.random.random(shape).astype(dtype)
        elif dtype == 'float64':
            return np.random.random(shape).astype(dtype)
        elif dtype == 'int32':
            return np.random.randint(
                self._low_data_bound, self._high_data_bound, shape
            ).astype(dtype)
        elif dtype == 'int64':
            return np.random.randint(
                self._low_data_bound, self._high_data_bound, shape
            ).astype(dtype)

    def _get_data(
        self, name, shape, dtype, set_feed_dict=True, append_batch_size=True
    ):
        if base.enabled():
            return base.to_variable(
                value=self._get_np_data(shape, dtype, append_batch_size),
                name=name,
                zero_copy=False,
            )
        else:
            if set_feed_dict:
                self._feed_dict[name] = self._get_np_data(
                    shape, dtype, append_batch_size
                )
            return layers.data(
                name=name,
                shape=shape,
                dtype=dtype,
                append_batch_size=append_batch_size,
            )

    def make_fit_a_line(self):
        with program_guard(
            fluid.default_main_program(),
            startup_program=fluid.default_startup_program(),
        ):
            x = self._get_data(name='x', shape=[13], dtype='float32')
            y_predict = layers.fc(input=x, size=1, act=None)
            y = self._get_data(name='y', shape=[1], dtype='float32')
            cost = paddle.nn.functional.square_error_cost(
                input=y_predict, label=y
            )
            avg_cost = paddle.mean(cost)
            return avg_cost

    def make_recognize_digits_mlp(self):
        with program_guard(
            fluid.default_main_program(), fluid.default_startup_program()
        ):
            # Change g_program, so the rest layers use `g_program`
            images = self._get_data(name='pixel', shape=[784], dtype='float32')
            label = self._get_data(name='label', shape=[1], dtype='int64')
            hidden1 = layers.fc(input=images, size=128, act='relu')
            hidden2 = layers.fc(input=hidden1, size=64, act='relu')
            predict = layers.fc(
                input=[hidden2, hidden1],
                size=10,
                act='softmax',
                param_attr=["sftmax.w1", "sftmax.w2"],
            )
            cost = paddle.nn.functional.cross_entropy(
                input=predict, label=label, reduction='none', use_softmax=False
            )
            avg_cost = paddle.mean(cost)
            return avg_cost

    def make_conv2d_transpose(self):
        with program_guard(
            fluid.default_main_program(), fluid.default_startup_program()
        ):
            img = self._get_data(name='pixel', shape=[3, 2, 2], dtype='float32')
            return paddle.static.nn.conv2d_transpose(
                input=img, num_filters=10, output_size=28
            )

    def make_recognize_digits_conv(self):
        with program_guard(
            fluid.default_main_program(), fluid.default_startup_program()
        ):
            images = self._get_data(
                name='pixel', shape=[1, 28, 28], dtype='float32'
            )
            label = self._get_data(name='label', shape=[1], dtype='int64')
            conv_pool_1 = nets.simple_img_conv_pool(
                input=images,
                filter_size=5,
                num_filters=2,
                pool_size=2,
                pool_stride=2,
                act="relu",
            )
            conv_pool_2 = nets.simple_img_conv_pool(
                input=conv_pool_1,
                filter_size=5,
                num_filters=4,
                pool_size=2,
                pool_stride=2,
                act="relu",
            )

            predict = layers.fc(input=conv_pool_2, size=10, act="softmax")
            cost = paddle.nn.functional.cross_entropy(
                input=predict, label=label, reduction='none', use_softmax=False
            )
            avg_cost = paddle.mean(cost)
            return avg_cost

    def make_word_embedding(self):
        with program_guard(
            fluid.default_main_program(), fluid.default_startup_program()
        ):
            dict_size = 10000
            embed_size = 32
            first_word = self._get_data(name='firstw', shape=[1], dtype='int64')
            second_word = self._get_data(
                name='secondw', shape=[1], dtype='int64'
            )
            third_word = self._get_data(name='thirdw', shape=[1], dtype='int64')
            forth_word = self._get_data(name='forthw', shape=[1], dtype='int64')
            next_word = self._get_data(name='nextw', shape=[1], dtype='int64')

            embed_first = layers.embedding(
                input=first_word,
                size=[dict_size, embed_size],
                dtype='float32',
                param_attr='shared_w',
            )
            embed_second = layers.embedding(
                input=second_word,
                size=[dict_size, embed_size],
                dtype='float32',
                param_attr='shared_w',
            )

            embed_third = layers.embedding(
                input=third_word,
                size=[dict_size, embed_size],
                dtype='float32',
                param_attr='shared_w',
            )
            embed_forth = layers.embedding(
                input=forth_word,
                size=[dict_size, embed_size],
                dtype='float32',
                param_attr='shared_w',
            )

            concat_embed = layers.concat(
                input=[embed_first, embed_second, embed_third, embed_forth],
                axis=1,
            )

            hidden1 = layers.fc(input=concat_embed, size=256, act='sigmoid')
            predict_word = layers.fc(
                input=hidden1, size=dict_size, act='softmax'
            )
            cost = paddle.nn.functional.cross_entropy(
                input=predict_word,
                label=next_word,
                reduction='none',
                use_softmax=False,
            )
            avg_cost = paddle.mean(cost)
            return avg_cost

    def make_pool2d(self):
        with program_guard(
            fluid.default_main_program(), fluid.default_startup_program()
        ):
            x = self._get_data(name='x', shape=[3, 224, 224], dtype='float32')
            return paddle.nn.functional.max_pool2d(
                x, kernel_size=[5, 3], stride=[1, 2], padding=(2, 1)
            )

    def make_pool2d_infershape(self):
        with program_guard(
            fluid.default_main_program(), fluid.default_startup_program()
        ):
            theta = self._get_data("theta", shape=[2, 3], dtype='float32')
            x = paddle.nn.functional.affine_grid(
                theta, out_shape=[2, 3, 244, 244]
            )
            return paddle.nn.functional.max_pool2d(
                x, kernel_size=[5, 3], stride=[1, 2], padding=(2, 1)
            )

    def make_softmax(self):
        with program_guard(
            fluid.default_main_program(), fluid.default_startup_program()
        ):
            data = self._get_data(name='data', shape=[10], dtype='float32')
            hid = layers.fc(input=data, size=20)
            return paddle.nn.functional.softmax(hid, axis=1)

    @prog_scope()
    def make_nce(self):
        window_size = 5
        words = []
        for i in range(window_size):
            words.append(
                self._get_data(
                    name='word_{0}'.format(i), shape=[1], dtype='int64'
                )
            )

        dict_size = 10000
        label_word = int(window_size // 2) + 1

        embs = []
        for i in range(window_size):
            if i == label_word:
                continue

            emb = layers.embedding(
                input=words[i],
                size=[dict_size, 32],
                param_attr='emb.w',
                is_sparse=True,
            )

            embs.append(emb)

        embs = layers.concat(input=embs, axis=1)
        loss = paddle.static.nn.nce(
            input=embs,
            label=words[label_word],
            num_total_classes=dict_size,
            param_attr='nce.w',
            bias_attr='nce.b',
        )
        avg_loss = paddle.mean(loss)
        return avg_loss

    def make_multiplex(self):
        with program_guard(
            fluid.default_main_program(), fluid.default_startup_program()
        ):
            x1 = self._get_data(name='x1', shape=[4], dtype='float32')
            x2 = self._get_data(name='x2', shape=[4], dtype='float32')
            index = self._get_data(name='index', shape=[1], dtype='int32')
            out = paddle.multiplex(inputs=[x1, x2], index=index)
            return out

    def make_softmax_with_cross_entropy(self):
        with program_guard(
            fluid.default_main_program(), fluid.default_startup_program()
        ):
            x = self._get_data(name='x', shape=[16], dtype='float32')
            y = self._get_data(name='label', shape=[1], dtype='int64')
            loss, softmax = paddle.nn.functional.softmax_with_cross_entropy(
                x, y, return_softmax=True
            )
            self.assertIsNotNone(loss)
            self.assertIsNotNone(softmax)

            loss = paddle.nn.functional.softmax_with_cross_entropy(x, y)
            self.assertIsNotNone(loss)

            x1 = self._get_data(name='x1', shape=[16, 32, 64], dtype='float32')
            y1 = self._get_data(name='label1', shape=[1, 32, 64], dtype='int64')
            y2 = self._get_data(name='label2', shape=[16, 1, 64], dtype='int64')
            y3 = self._get_data(name='label3', shape=[16, 32, 1], dtype='int64')
            loss1 = paddle.nn.functional.softmax_with_cross_entropy(
                x1, y1, axis=1
            )
            loss2 = paddle.nn.functional.softmax_with_cross_entropy(
                x1, y2, axis=2
            )
            loss3 = paddle.nn.functional.softmax_with_cross_entropy(
                x1, y3, axis=3
            )
            loss4 = paddle.nn.functional.softmax_with_cross_entropy(
                x1, y3, axis=-1
            )
            self.assertIsNotNone(loss1)
            self.assertIsNotNone(loss2)
            self.assertIsNotNone(loss3)
            self.assertIsNotNone(loss4)
            return loss4

    def make_scatter(self):
        with program_guard(
            fluid.default_main_program(), fluid.default_startup_program()
        ):
            x = self._get_data(
                name='x', shape=[3, 3], append_batch_size=False, dtype='float32'
            )
            idx = self._get_data(
                name='idx', shape=[2], append_batch_size=False, dtype='int32'
            )
            updates = self._get_data(
                name='updates',
                shape=[2, 3],
                append_batch_size=False,
                dtype='float32',
            )
            out = paddle.scatter(x, index=idx, updates=updates)
            return out

    def make_one_hot(self):
        with fluid.framework._dygraph_place_guard(place=fluid.CPUPlace()):
            label = self._get_data(name="label", shape=[1], dtype="int32")
            one_hot_label = layers.one_hot(input=label, depth=10)
            return one_hot_label

    def make_label_smooth(self):
        # TODO(minqiyang): support gpu ut
        self._force_to_use_cpu = True
        with fluid.framework._dygraph_place_guard(place=fluid.CPUPlace()):
            label = self._get_data(name="label", shape=[1], dtype="int32")
            one_hot_label = layers.one_hot(input=label, depth=10)
            smooth_label = F.label_smooth(label=one_hot_label, epsilon=0.1)
            return smooth_label

    def make_topk(self):
        with program_guard(
            fluid.default_main_program(), fluid.default_startup_program()
        ):
            data = self._get_data(name="label", shape=[200], dtype="float32")
            values, indices = paddle.topk(data, k=5)
            return values
            return indices

    def make_l2_normalize(self):
        with program_guard(
            fluid.default_main_program(), fluid.default_startup_program()
        ):
            x = self._get_data(name='x', shape=[8, 7, 10], dtype="float32")
            output = paddle.nn.functional.normalize(x, axis=1)
            return output

    def make_shape(self):
        with program_guard(
            fluid.default_main_program(), fluid.default_startup_program()
        ):
            input = self._get_data(
                name="input", shape=[3, 100, 100], dtype="float32"
            )
            out = paddle.shape(input)
            return out

    def make_pad2d(self):
        with program_guard(
            fluid.default_main_program(), fluid.default_startup_program()
        ):
            input = self._get_data(
                name="input", shape=[3, 100, 100], dtype="float32"
            )

            tmp_pad = paddle.nn.Pad2D(
                padding=[1, 2, 3, 4],
                mode='reflect',
                data_format='NCHW',
                name="shape",
            )
            out = tmp_pad(input)
            return out

    def make_mish(self):
        with program_guard(
            fluid.default_main_program(), fluid.default_startup_program()
        ):
            input = self._get_data(name="input", shape=[16], dtype="float32")
            out = paddle.nn.functional.mish(input, name='mish')
            return out

    def make_cross_entropy(self):
        with program_guard(
            fluid.default_main_program(), fluid.default_startup_program()
        ):
            x = self._get_data(name="x", shape=[30, 10], dtype="float32")
            label = self._get_data(name="label", shape=[30, 1], dtype="int64")
            mode = 'channel'
            out = paddle.nn.functional.cross_entropy(
                x,
                label,
                soft_label=False,
                ignore_index=4,
                reduction='none',
                use_softmax=False,
            )
            return out

    def make_uniform_random_batch_size_like(self):
        with program_guard(
            fluid.default_main_program(), fluid.default_startup_program()
        ):
            input = self._get_data(
                name="input", shape=[13, 11], dtype='float32'
            )
            out = random.uniform_random_batch_size_like(input, [-1, 11])
            return out

    def make_gaussian_random(self):
        with program_guard(
            fluid.default_main_program(), fluid.default_startup_program()
        ):
            out = random.gaussian(shape=[20, 30])
            return out

    def make_sum(self):
        with program_guard(
            fluid.default_main_program(), fluid.default_startup_program()
        ):
            input = self._get_data(
                name="input", shape=[13, 11], dtype='float32'
            )

            out = paddle.add_n(input)
            return out

    def make_slice(self):
        starts = [1, 0, 2]
        ends = [3, 3, 4]
        axes = [0, 1, 2]

        with program_guard(
            fluid.default_main_program(), fluid.default_startup_program()
        ):
            input = self._get_data(
                name="input", shape=[3, 4, 5, 6], dtype='float32'
            )

            out = paddle.slice(input, axes=axes, starts=starts, ends=ends)
            return out

    def make_scale_variable(self):
        with program_guard(
            fluid.default_main_program(), fluid.default_startup_program()
        ):
            input = self._get_data(
                name="input", shape=[3, 4, 5, 6], dtype='float32'
            )
            scale_var = self._get_data(
                name="scale",
                shape=[1],
                dtype='float32',
                append_batch_size=False,
            )
            out = paddle.scale(input, scale=scale_var)
            return out

    def make_bilinear_tensor_product_layer(self):
        with program_guard(
            fluid.default_main_program(), fluid.default_startup_program()
        ):
            data = self._get_data(name='data', shape=[4], dtype="float32")

            theta = self._get_data(name="theta", shape=[5], dtype="float32")
            out = paddle.static.nn.common.bilinear_tensor_product(
                data, theta, 6
            )
            return out

    def make_batch_norm(self):
        with program_guard(
            fluid.default_main_program(), fluid.default_startup_program()
        ):
            data = self._get_data(
                name='data', shape=[32, 128, 128], dtype="float32"
            )
            out = paddle.static.nn.batch_norm(data)
            return out

    def make_batch_norm_momentum_variable(self):
        with program_guard(
            fluid.default_main_program(), fluid.default_startup_program()
        ):
            data = self._get_data(
                name='data', shape=[32, 128, 128], dtype="float32"
            )
            momentum = self._get_data(
                name='momentum',
                shape=[1],
                dtype='float32',
                append_batch_size=False,
            )
            out = paddle.static.nn.batch_norm(data, momentum=momentum)
            return out

    def make_range(self):
        with program_guard(
            fluid.default_main_program(), fluid.default_startup_program()
        ):
            paddle.arange(0, 10, 2, 'int32')
            paddle.arange(0.1, 10.0, 0.2, 'float32')
            paddle.arange(0.1, 10.0, 0.2, 'float64')
            start = layers.fill_constant(shape=[1], value=0.1, dtype="float32")
            end = layers.fill_constant(shape=[1], value=10.0, dtype="float32")
            step = layers.fill_constant(shape=[1], value=0.2, dtype="float32")
            y = paddle.arange(start, end, step, 'float64')
            return y

    def make_spectral_norm(self):
        with program_guard(
            fluid.default_main_program(), fluid.default_startup_program()
        ):
            weight = self._get_data(
                name='weight',
                shape=[2, 3, 32, 32],
                dtype="float32",
                append_batch_size=False,
            )
            out = layers.spectral_norm(weight, dim=1, power_iters=1)
            return out

    def make_kldiv_loss(self):
        with program_guard(
            fluid.default_main_program(), fluid.default_startup_program()
        ):
            x = self._get_data(
                name='x',
                shape=[32, 128, 128],
                dtype="float32",
                append_batch_size=False,
            )
            target = self._get_data(
                name='target',
                shape=[32, 128, 128],
                dtype="float32",
                append_batch_size=False,
            )
            loss = paddle.nn.functional.kl_div(
                input=x, label=target, reduction='batchmean'
            )
            return loss

    def make_pixel_shuffle(self):
        with program_guard(
            fluid.default_main_program(), fluid.default_startup_program()
        ):
            x = self._get_data(name="X", shape=[9, 4, 4], dtype="float32")
            out = paddle.nn.functional.pixel_shuffle(x, upscale_factor=3)
            return out

    def make_mse_loss(self):
        with program_guard(
            fluid.default_main_program(), fluid.default_startup_program()
        ):
            x = self._get_data(name="X", shape=[1], dtype="float32")
            y = self._get_data(name="Y", shape=[1], dtype="float32")
            out = paddle.nn.functional.mse_loss(input=x, label=y)
            return out

    def make_square_error_cost(self):
        with program_guard(
            fluid.default_main_program(), fluid.default_startup_program()
        ):
            x = self._get_data(name="X", shape=[1], dtype="float32")
            y = self._get_data(name="Y", shape=[1], dtype="float32")
            out = paddle.nn.functional.square_error_cost(input=x, label=y)
            return out

    def test_lod_reset(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            # case 1
            x = layers.data(name='x', shape=[10], dtype='float32')
            y = layers.data(
                name='y', shape=[10, 20], dtype='float32', lod_level=2
            )
            z = layers.lod_reset(x=x, y=y)
            self.assertTrue(z.lod_level == 2)
            # case 2
            lod_tensor_in = layers.data(name='lod_in', shape=[1], dtype='int32')
            z = layers.lod_reset(x=x, y=lod_tensor_in)
            self.assertTrue(z.lod_level == 1)
            # case 3
            z = layers.lod_reset(x=x, target_lod=[1, 2, 3])
            self.assertTrue(z.lod_level == 1)
            return z

    def test_affine_grid(self):
        with self.static_graph():
            data = layers.data(name='data', shape=[2, 3, 3], dtype="float32")
            out = paddle.argsort(x=data, axis=1)

            theta = layers.data(name="theta", shape=[2, 3], dtype="float32")
            out_shape = layers.data(name="out_shape", shape=[-1], dtype="int32")
            data_0 = paddle.nn.functional.affine_grid(theta, out_shape)
            data_1 = paddle.nn.functional.affine_grid(theta, [5, 3, 28, 28])

            self.assertIsNotNone(data_0)
            self.assertIsNotNone(data_1)

    def test_stridedslice(self):
        axes = [0, 1, 2]
        starts = [1, 0, 2]
        ends = [3, 3, 4]
        strides = [1, 1, 1]
        with self.static_graph():
            x = layers.data(name="x", shape=[245, 30, 30], dtype="float32")
            out = paddle.strided_slice(
                x, axes=axes, starts=starts, ends=ends, strides=strides
            )
            return out

    def test_fill_constant_batch_size_like(self):
        with self.static_graph():
            like = fluid.layers.fill_constant(
                shape=[1, 200], value=10, dtype='int64'
            )
            out = layers.fill_constant_batch_size_like(
                input=like, shape=[2, 3300], value=1315454564656, dtype='int64'
            )
            return out

    def test_sequence_expand(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            x = layers.data(name='x', shape=[10], dtype='float32')
            y = layers.data(
                name='y', shape=[10, 20], dtype='float32', lod_level=2
            )
            return layers.sequence_expand(x=x, y=y, ref_level=1)

    def test_sequence_reshape(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            x = layers.data(name='x', shape=[8], dtype='float32', lod_level=1)
            out = layers.sequence_reshape(input=x, new_dim=16)
            return out

    def test_sequence_unpad(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            x = layers.data(name='x', shape=[10, 5], dtype='float32')
            length = layers.data(name='length', shape=[], dtype='int64')
            return layers.sequence_unpad(x=x, length=length)

    def test_sequence_softmax(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            seq_data = layers.data(
                name='seq_data', shape=[10, 10], dtype='float32', lod_level=1
            )
            seq = layers.fc(input=seq_data, size=20)
            return layers.sequence_softmax(seq)

    def test_sequence_unsqueeze(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            x = layers.data(name='x', shape=[8, 2], dtype='float32')
            out = paddle.unsqueeze(x, axis=[1])
            return out

    def test_sequence_scatter(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            x = layers.data(
                name='x', shape=[3, 6], append_batch_size=False, dtype='float32'
            )
            idx = layers.data(
                name='idx',
                shape=[12, 1],
                append_batch_size=False,
                dtype='int32',
                lod_level=1,
            )
            updates = layers.data(
                name='updates',
                shape=[12, 1],
                append_batch_size=False,
                dtype='float32',
                lod_level=1,
            )
            out = layers.sequence_scatter(input=x, index=idx, updates=updates)
            return out

    def test_sequence_slice(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            import numpy as np

            seqs = layers.data(
                name='x', shape=[10, 5], dtype='float32', lod_level=1
            )
            offset = layers.assign(input=np.array([[0, 1]]).astype('int32'))
            length = layers.assign(input=np.array([[2, 1]]).astype('int32'))
            out = layers.sequence_slice(
                input=seqs, offset=offset, length=length
            )
            return out

    def test_shuffle_batch(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            x = layers.data(
                name='X', shape=[4, 50], dtype='float32', lod_level=0
            )
            out1 = fluid.contrib.layers.shuffle_batch(x)
            default_main_program().random_seed = 1000
            out2 = fluid.contrib.layers.shuffle_batch(x)
            self.assertIsNotNone(out1)
            self.assertIsNotNone(out2)
            return out1

    def test_partial_sum(self):
        with self.static_graph():
            x = fluid.data(name="x", shape=[None, 3], dtype="float32")
            y = fluid.data(name="y", shape=[None, 3], dtype="float32")
            sum = fluid.contrib.layers.partial_sum(
                [x, y], start_index=0, length=2
            )
            return sum

    def test_batch_fc(self):
        with self.static_graph():
            input = fluid.data(name="input", shape=[16, 2, 3], dtype="float32")
            out = fluid.contrib.layers.batch_fc(
                input=input,
                param_size=[16, 3, 10],
                param_attr=fluid.ParamAttr(
                    learning_rate=1.0,
                    name="w_0",
                    initializer=fluid.initializer.Xavier(uniform=False),
                ),
                bias_size=[16, 10],
                bias_attr=fluid.ParamAttr(
                    learning_rate=1.0,
                    name="b_0",
                    initializer=fluid.initializer.Xavier(uniform=False),
                ),
                act="relu",
            )
        return out

    def test_rank_attention(self):
        with self.static_graph():
            input = fluid.data(name="input", shape=[None, 2], dtype="float32")
            rank_offset = fluid.data(
                name="rank_offset", shape=[None, 7], dtype="int32"
            )
            out = fluid.contrib.layers.rank_attention(
                input=input,
                rank_offset=rank_offset,
                rank_param_shape=[18, 3],
                rank_param_attr=fluid.ParamAttr(
                    learning_rate=1.0,
                    name="ubm_rank_param.w_0",
                    initializer=fluid.initializer.Xavier(uniform=False),
                ),
                max_rank=3,
            )
            return out

    def test_sequence_enumerate(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            x = layers.data(name="input", shape=[1], dtype='int32', lod_level=1)
            out = layers.sequence_enumerate(input=x, win_size=2, pad_value=0)

    def test_row_conv(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            x = layers.data(name='x', shape=[16], dtype='float32', lod_level=1)
            out = layers.row_conv(input=x, future_context_size=2)
            return out

    def test_simple_conv2d(self):
        # TODO(minqiyang): dygraph do not support layers with param now
        with self.static_graph():
            images = layers.data(
                name='pixel', shape=[3, 48, 48], dtype='float32'
            )
            return layers.conv2d(
                input=images, num_filters=3, filter_size=[4, 4]
            )

    def test_squeeze(self):
        # TODO(minqiyang): dygraph do not support layers with param now
        with self.static_graph():
            x = layers.data(name='x', shape=[1, 1, 4], dtype='float32')
            out = paddle.squeeze(x, axis=[2])
            return out

    def test_flatten(self):
        # TODO(minqiyang): dygraph do not support op without kernel now
        with self.static_graph():
            x = layers.data(
                name='x',
                append_batch_size=False,
                shape=[4, 4, 3],
                dtype="float32",
            )
            out = paddle.flatten(x, 1, -1, name="flatten")
            return out

    def test_linspace(self):
        program = Program()
        with program_guard(program):
            out = paddle.linspace(20, 10, 5, 'float64')
            self.assertIsNotNone(out)
        print(str(program))

    def test_unfold(self):
        with self.static_graph():
            x = layers.data(name='x', shape=[3, 20, 20], dtype='float32')
            out = paddle.nn.functional.unfold(x, [3, 3], 1, 1, 1)
            return out

    def test_partial_concat(self):
        with self.static_graph():
            x = fluid.data(name="x", shape=[None, 3], dtype="float32")
            y = fluid.data(name="y", shape=[None, 3], dtype="float32")
            concat1 = fluid.contrib.layers.partial_concat(
                [x, y], start_index=0, length=2
            )
            concat2 = fluid.contrib.layers.partial_concat(
                x, start_index=0, length=-1
            )
            return concat1, concat2

    def test_addmm(self):
        with program_guard(
            fluid.default_main_program(), fluid.default_startup_program()
        ):
            input = layers.data(
                name='input_data',
                shape=[3, 3],
                append_batch_size=False,
                dtype='float32',
            )
            x = layers.data(
                name='x', shape=[3, 2], append_batch_size=False, dtype='float32'
            )
            y = layers.data(
                name='y', shape=[2, 3], append_batch_size=False, dtype='float32'
            )

            out = paddle.addmm(input=input, x=x, y=y)
            return out

    def test_warpctc_with_padding(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            input_length = paddle.static.data(
                name='logits_length', shape=[11], dtype='int64'
            )
            label_length = paddle.static.data(
                name='labels_length', shape=[12], dtype='int64'
            )
            label = paddle.static.data(
                name='label', shape=[12, 1], dtype='int32'
            )
            predict = paddle.static.data(
                name='predict', shape=[4, 4, 8], dtype='float32'
            )
            output = paddle.nn.functional.ctc_loss(
                log_probs=predict,
                labels=label,
                input_lengths=input_length,
                label_lengths=label_length,
                reduction='none',
            )
            return output

    def test_basic_gru(self):
        input_size = 128
        hidden_size = 256
        with self.static_graph():
            input = fluid.data(
                name="input", shape=[None, None, input_size], dtype='float32'
            )
            pre_hidden = fluid.data(
                name="pre_hidden", shape=[None, hidden_size], dtype='float32'
            )
            sequence_length = fluid.data(
                name="sequence_length", shape=[None], dtype='int32'
            )

            for bidirectional in [True, False]:
                for batch_first in [True, False]:
                    rnn_out, last_hidden = fluid.contrib.layers.basic_gru(
                        input,
                        pre_hidden,
                        hidden_size=256,
                        num_layers=2,
                        sequence_length=sequence_length,
                        dropout_prob=0.5,
                        bidirectional=bidirectional,
                        batch_first=batch_first,
                    )


class ExampleNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[1, 1], attr=paddle.ParamAttr(trainable=False)
        )

    def forward(self):
        # only for test parameter trainable attr
        pass


class TestLayerParameterTrainableSet(unittest.TestCase):
    def test_layer_parameter_set(self):
        with fluid.dygraph.guard():
            net = ExampleNet()
            self.assertFalse(net.weight.trainable)


class TestLayerTrainingAttribute(unittest.TestCase):
    def test_set_train_eval_in_dynamic_mode(self):
        with fluid.dygraph.guard():
            net = paddle.nn.Dropout()
            net.train()
            self.assertTrue(net.training)
            net.eval()
            self.assertFalse(net.training)

    def test_set_train_eval_in_static_mode(self):
        net = paddle.nn.Dropout()
        net.train()
        self.assertTrue(net.training)
        net.eval()
        self.assertFalse(net.training)


class MyLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self._linear = paddle.nn.Linear(1, 1)
        self._dropout = paddle.nn.Dropout(p=0.5)

    def forward(self, input):
        temp = self._linear(input)
        temp = self._dropout(temp)
        return temp


class MySuperLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self._mylayer = MyLayer()

    def forward(self, input):
        temp = self._mylayer(input)
        return temp


class TestSubLayerCount(unittest.TestCase):
    def test_sublayer(self):
        with fluid.dygraph.guard():
            mySuperlayer = MySuperLayer()
            self.assertTrue(len(mySuperlayer.sublayers()) == 3)
            self.assertTrue(len(mySuperlayer.sublayers(include_self=True)) == 4)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
