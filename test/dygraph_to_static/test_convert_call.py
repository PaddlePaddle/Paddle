#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import logging
import unittest

import numpy as np
from dygraph_to_static_util import ast_only_test, dy2static_unittest

import paddle
import paddle.jit.dy2static as _jst
from paddle import base
from paddle.jit.dy2static.convert_call_func import CONVERSION_OPTIONS
from paddle.jit.dy2static.utils import func_to_source_code

SEED = 2020
np.random.seed(SEED)

# Situation 1 : test recursive call


# Use a decorator to test exception
@paddle.jit.to_static
def dyfunc_with_if(x_v):
    if paddle.mean(x_v).numpy() > 5:
        x_v = x_v - 1
    else:
        x_v = x_v + 1
    return x_v


@paddle.jit.to_static
def nested_func(x_v):
    x_v = base.dygraph.to_variable(x_v)

    def fn1():
        return x_v

    res = fn1()
    return res


@paddle.jit.to_static
def dyfunc_with_third_library_logging(x_v):
    logging.info('test dyfunc_with_third_library_logging')
    if paddle.mean(x_v).numpy() > 5:
        x_v = x_v - 1
    else:
        x_v = x_v + 1
    return x_v


class A:
    @staticmethod
    def add(a, b):
        """
        dygraph mode, return a numpy object.
        static graph mode, return a variable object.
        """
        return paddle.to_tensor(a.numpy() + b.numpy())


@paddle.jit.to_static
def dyfunc_with_staticmethod(x_v):
    a = A()
    return a.add(x_v, x_v)


class TestRecursiveCall1(unittest.TestCase):
    def setUp(self):
        self.input = np.random.random([10, 16]).astype('float32')
        self.place = (
            base.CUDAPlace(0)
            if base.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        self.init_test_func()

    def init_test_func(self):
        self.dyfunc = nested_func

    def get_dygraph_output(self):
        paddle.jit.enable_to_static(False)
        with base.dygraph.guard():
            res = self.dyfunc(self.input).numpy()
            return res

    def get_static_output(self):
        paddle.jit.enable_to_static(True)
        with base.dygraph.guard():
            res = self.dyfunc(self.input).numpy()
            return res

    def test_transformed_static_result(self):
        static_res = self.get_static_output()
        dygraph_res = self.get_dygraph_output()
        np.testing.assert_allclose(
            dygraph_res,
            static_res,
            rtol=1e-05,
            err_msg='dygraph res is {}\nstatic_res is {}'.format(
                dygraph_res, static_res
            ),
        )


lambda_fun = lambda x: x


class MyConvLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self._conv = paddle.nn.Conv2D(
            in_channels=3,
            out_channels=2,
            kernel_size=3,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.99)
            ),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.5)
            ),
        )

    @paddle.jit.to_static
    def forward(self, inputs):
        y = dyfunc_with_if(inputs)
        y = lambda_fun(y)
        y = self.dymethod(y)
        return y

    @paddle.jit.to_static
    def dymethod(self, x_v):
        x_v = paddle.assign(x_v)
        return x_v


class MyLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

        self.conv = MyConvLayer()
        self.fc = paddle.nn.Linear(
            in_features=5,
            out_features=1,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.99)
            ),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.5)
            ),
        )
        self.act = paddle.nn.ReLU()

    @paddle.jit.to_static
    def forward(self, inputs):
        h = self.conv(inputs)
        out = self.fc(h)
        return self.act(out)


class TestRecursiveCall2(unittest.TestCase):
    def setUp(self):
        self.input = np.random.random((1, 3, 3, 5)).astype('float32')
        self.place = (
            base.CUDAPlace(0)
            if base.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        self.set_func()

    def set_func(self):
        self.dygraph_func = MyLayer()

    def _run(self):
        with base.dygraph.guard():
            data = base.dygraph.to_variable(self.input)
            res = self.dygraph_func(data)

            return res.numpy()

    def get_dygraph_output(self):
        paddle.jit.enable_to_static(False)
        return self._run()

    def get_static_output(self):
        paddle.jit.enable_to_static(True)
        return self._run()

    def test_transformed_static_result(self):
        dygraph_res = self.get_dygraph_output()
        static_res = self.get_static_output()
        np.testing.assert_allclose(dygraph_res, static_res, rtol=1e-05)


class TestThirdPartyLibrary(TestRecursiveCall2):
    def set_func(self):
        self.dygraph_func = dyfunc_with_third_library_logging


class TestStaticMethod(TestRecursiveCall2):
    def set_func(self):
        self.dygraph_func = dyfunc_with_staticmethod


# Situation 2 : test not_to_static


class NotToStaticHelper(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def sum(self, x):
        if x.shape[0] > 1:
            res = x + 1
        res = paddle.sum(x)
        return res

    def outer(self, x):
        res = self.sum(x)
        return res

    def inner(self, x):
        return self.outer(x)


class TestNotToConvert(TestRecursiveCall2):
    def set_func(self):
        self.net = NotToStaticHelper()
        paddle.jit.not_to_static(self.net.sum)
        self.dygraph_func = paddle.jit.to_static(self.net.outer)

    def test_conversion_options(self):
        options = getattr(self.net.sum, CONVERSION_OPTIONS, None)
        self.assertIsNotNone(options)
        self.assertTrue(options.not_convert)

    def test_code(self):
        # check 'if statement' is not converted
        self.assertIn(
            "if x.shape[0] > 1", func_to_source_code(_jst.Call(self.net.sum))
        )


@dy2static_unittest
class TestNotToConvert2(TestRecursiveCall2):
    def set_func(self):
        self.net = NotToStaticHelper()
        # for to_static(not_to_static(function))  == enable_static
        paddle.jit.not_to_static(self.net.sum)
        self.dygraph_func = paddle.jit.to_static(self.net.sum)

    def test_conversion_options(self):
        options = getattr(self.net.sum, CONVERSION_OPTIONS, None)
        self.assertIsNotNone(options)
        self.assertTrue(options.not_convert)

    @ast_only_test
    def test_code(self):
        self.dygraph_func = paddle.jit.to_static(self.net.sum)
        # check 'if statement' is not converted
        self.assertIn("if x.shape[0] > 1", self.dygraph_func.code)


# Situation 3 : test to_static for paddle api
@paddle.jit.not_to_static
def forward(self, x):
    if x.shape[0] > 1:
        x = x + 1
    return x


@dy2static_unittest
class TestConvertPaddleAPI(unittest.TestCase):
    @ast_only_test
    def test_functional_api(self):
        func = paddle.nn.functional.relu
        func = paddle.jit.to_static(func)
        self.assertNotIn("_jst.IfElse", func.code)
        self.assertIn("if in_dynamic_mode()", func.code)

    @ast_only_test
    def test_class_api(self):
        bn = paddle.nn.SyncBatchNorm(2)
        paddle.jit.to_static(bn)
        self.assertNotIn("_jst.IfElse", bn.forward.code)
        self.assertIn("if in_dynamic_mode()", bn.forward.code)

    @ast_only_test
    def test_class_patch_api(self):
        paddle.nn.SyncBatchNorm.forward = forward
        bn = paddle.nn.SyncBatchNorm(2)
        paddle.jit.to_static(bn)
        self.assertNotIn("_jst.IfElse", bn.forward.code)
        self.assertIn("if x.shape[0] > 1", bn.forward.code)


if __name__ == '__main__':
    unittest.main()
