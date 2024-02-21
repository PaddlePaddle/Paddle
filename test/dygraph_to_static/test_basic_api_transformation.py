#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from dygraph_to_static_utils import (
    Dy2StTestBase,
    test_default_and_pir,
)

import paddle
from paddle import to_tensor
from paddle.jit.api import to_static

SEED = 2020
np.random.seed(SEED)


def dyfunc_to_tensor(x):
    res1 = paddle.to_tensor(x, dtype=None, place=None, stop_gradient=True)
    res2 = paddle.tensor.to_tensor(data=res1)
    res3 = to_tensor(data=res2)
    return res3


def dyfunc_int_to_tensor(x):
    res = paddle.to_tensor(3)
    return res


def dyfunc_float_to_tensor(x):
    return paddle.to_tensor(2.0)


def dyfunc_bool_to_tensor(x):
    return paddle.to_tensor(True)


class TestDygraphBasicApi_ToVariable(Dy2StTestBase):
    def setUp(self):
        self.input = np.ones(5).astype("int32")
        self.test_funcs = [
            dyfunc_to_tensor,
            dyfunc_bool_to_tensor,
            dyfunc_int_to_tensor,
            dyfunc_float_to_tensor,
        ]

    def get_dygraph_output(self):
        res = self.dygraph_func(self.input).numpy()
        return res

    def get_static_output(self):
        static_res = to_static(self.dygraph_func)(self.input).numpy()

        return static_res

    @test_default_and_pir
    def test_transformed_static_result(self):
        for func in self.test_funcs:
            self.dygraph_func = func
            dygraph_res = self.get_dygraph_output()
            static_res = self.get_static_output()
            np.testing.assert_allclose(dygraph_res, static_res, rtol=1e-05)


# test Apis that inherit from layers.Layer
def dyfunc_BilinearTensorProduct(bilinearTensorProduct, x1, x2):
    res = bilinearTensorProduct(
        paddle.to_tensor(x1),
        paddle.to_tensor(x2),
    )
    return res


def dyfunc_conv2d(conv2d, input):
    res = conv2d(input)
    return res


def dyfunc_conv3d(conv3d, input):
    res = conv3d(input)
    return res


def dyfunc_conv2d_transpose(conv2dTranspose, input):
    ret = conv2dTranspose(input)
    return ret


def dyfunc_conv3d_transpose(conv3dTranspose, input):
    ret = conv3dTranspose(input)
    return ret


def dyfunc_linear(fc, m, input):
    res = fc(input)
    return m(res)


def dyfunc_pool2d(input):
    paddle.nn.AvgPool2D(kernel_size=2, stride=1)
    pool2d = paddle.nn.AvgPool2D(kernel_size=2, stride=1)
    res = pool2d(input)
    return res


def dyfunc_prelu(prelu0, input):
    res = prelu0(input)
    return res


class TestDygraphBasicApi(Dy2StTestBase):
    # Compare results of dynamic graph and transformed static graph function which only
    # includes basic Api.

    def setUp(self):
        self.input = np.random.random((1, 4, 3, 3)).astype('float32')
        self.dygraph_func = dyfunc_pool2d

    def get_dygraph_output(self):
        paddle.seed(SEED)
        data = paddle.to_tensor(self.input)
        res = self.dygraph_func(data).numpy()

        return res

    def get_static_output(self):
        data = paddle.assign(self.input)
        static_res = to_static(self.dygraph_func)(data).numpy()

        return static_res

    @test_default_and_pir
    def test_transformed_static_result(self):
        dygraph_res = self.get_dygraph_output()
        static_res = self.get_static_output()
        np.testing.assert_allclose(dygraph_res, static_res, rtol=1e-05)


class TestDygraphBasicApi_BilinearTensorProduct(TestDygraphBasicApi):
    def setUp(self):
        self.input1 = np.random.random((5, 5)).astype('float32')
        self.input2 = np.random.random((5, 4)).astype('float32')

        bilinearTensorProduct = paddle.nn.Bilinear(
            5,
            4,
            1000,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.99)
            ),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.5)
            ),
        )

        self.dygraph_func = lambda x, y: dyfunc_BilinearTensorProduct(
            bilinearTensorProduct, x, y
        )

    def get_dygraph_output(self):
        paddle.seed(SEED)
        res = self.dygraph_func(self.input1, self.input2).numpy()
        return res

    def get_static_output(self):
        static_res = to_static(self.dygraph_func)(
            self.input1, self.input2
        ).numpy()

        return static_res


class TestDygraphBasicApi_Conv2D(TestDygraphBasicApi):
    def setUp(self):
        conv2d = paddle.nn.Conv2D(
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
        self.input = np.random.random((1, 3, 3, 5)).astype('float32')
        self.dygraph_func = lambda x: dyfunc_conv2d(conv2d, x)


class TestDygraphBasicApi_Conv3D(TestDygraphBasicApi):
    def setUp(self):
        self.input = np.random.random((1, 3, 3, 3, 5)).astype('float32')
        conv3d = paddle.nn.Conv3D(
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
        self.dygraph_func = lambda x: dyfunc_conv3d(conv3d, x)


class TestDygraphBasicApi_Conv2DTranspose(TestDygraphBasicApi):
    def setUp(self):
        self.input = np.random.random((5, 3, 32, 32)).astype('float32')
        conv2d_transpose = paddle.nn.Conv2DTranspose(
            3,
            12,
            12,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.99)
            ),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.5)
            ),
        )

        self.dygraph_func = lambda x: dyfunc_conv2d_transpose(
            conv2d_transpose, x
        )


class TestDygraphBasicApi_Conv3DTranspose(TestDygraphBasicApi):
    def setUp(self):
        self.input = np.random.random((5, 3, 12, 32, 32)).astype('float32')

        conv3d_transpose = paddle.nn.Conv3DTranspose(
            in_channels=3,
            out_channels=12,
            kernel_size=12,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.99)
            ),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.5)
            ),
        )

        self.dygraph_func = lambda x: dyfunc_conv3d_transpose(
            conv3d_transpose, x
        )


class TestDygraphBasicApi_Linear(TestDygraphBasicApi):
    def setUp(self):
        self.input = np.random.random((4, 3, 10)).astype('float32')
        fc = paddle.nn.Linear(
            in_features=10,
            out_features=5,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.99)
            ),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.5)
            ),
        )
        m = paddle.nn.ReLU()
        self.dygraph_func = lambda x: dyfunc_linear(fc, m, x)


class TestDygraphBasicApi_Prelu(TestDygraphBasicApi):
    def setUp(self):
        self.input = np.ones([5, 20, 10, 10]).astype('float32')
        prelu0 = paddle.nn.PReLU(
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(1.0)
            ),
        )
        self.dygraph_func = lambda x: dyfunc_prelu(prelu0, x)


# 2. test Apis that inherit from LearningRateDecay
def dyfunc_cosine_decay(CosineDecay):
    lr = CosineDecay()
    return paddle.to_tensor(lr)


def dyfunc_exponential_decay():
    base_lr = 0.1
    exponential_decay = paddle.optimizer.lr.ExponentialDecay(
        learning_rate=base_lr, gamma=0.5
    )
    lr = exponential_decay()
    return lr


def dyfunc_inverse_time_decay():
    base_lr = 0.1
    inverse_time_decay = paddle.optimizer.lr.InverseTimeDecay(
        learning_rate=base_lr, gamma=0.5
    )
    lr = inverse_time_decay()
    return lr


def dyfunc_natural_exp_decay():
    base_lr = 0.1
    natural_exp_decay = paddle.optimizer.lr.NaturalExpDecay(
        learning_rate=base_lr, gamma=0.5
    )
    lr = natural_exp_decay()
    return lr


def dyfunc_noam_decay():
    noam_decay = paddle.optimizer.lr.NoamDecay(100, 100)
    lr = noam_decay()
    return paddle.to_tensor(lr)


def dyfunc_piecewise_decay():
    boundaries = [10000, 20000]
    values = [1.0, 0.5, 0.1]
    pd = paddle.optimizer.lr.PiecewiseDecay(boundaries, values)
    lr = pd()
    return paddle.to_tensor(lr)


def dyfunc_polynomial_decay():
    start_lr = 0.01
    total_step = 5000
    end_lr = 0
    pd = paddle.optimizer.lr.PolynomialDecay(
        start_lr, total_step, end_lr, power=1.0
    )
    lr = pd()
    return paddle.to_tensor(lr)


class TestDygraphBasicApi_CosineDecay(Dy2StTestBase):
    def setUp(self):
        base_lr = 0.1
        CosineDecay = paddle.optimizer.lr.CosineAnnealingDecay(
            learning_rate=base_lr, T_max=120
        )
        self.dygraph_func = lambda: dyfunc_cosine_decay(CosineDecay)

    def get_dygraph_output(self):
        res = self.dygraph_func().numpy()
        return res

    def get_static_output(self):
        static_res = to_static(self.dygraph_func)()
        return static_res

    @test_default_and_pir
    def test_transformed_static_result(self):
        dygraph_res = self.get_dygraph_output()
        static_res = self.get_static_output()
        np.testing.assert_allclose(dygraph_res, static_res, rtol=1e-05)


class TestDygraphBasicApi_ExponentialDecay(TestDygraphBasicApi_CosineDecay):
    def setUp(self):
        self.dygraph_func = dyfunc_exponential_decay

    def get_dygraph_output(self):
        paddle.seed(SEED)
        res = self.dygraph_func()
        return res

    def get_static_output(self):
        static_out = to_static(self.dygraph_func)()

        return static_out


class TestDygraphBasicApi_InverseTimeDecay(TestDygraphBasicApi_CosineDecay):
    def setUp(self):
        self.dygraph_func = dyfunc_inverse_time_decay

    def get_dygraph_output(self):
        paddle.seed(SEED)
        res = self.dygraph_func()
        return res

    def get_static_output(self):
        static_out = to_static(self.dygraph_func)()

        return static_out


class TestDygraphBasicApi_NaturalExpDecay(TestDygraphBasicApi_CosineDecay):
    def setUp(self):
        self.dygraph_func = dyfunc_natural_exp_decay

    def get_dygraph_output(self):
        paddle.seed(SEED)
        res = self.dygraph_func()
        return res

    def get_static_output(self):
        static_out = to_static(self.dygraph_func)()

        return static_out


class TestDygraphBasicApi_NoamDecay(TestDygraphBasicApi_CosineDecay):
    def setUp(self):
        self.dygraph_func = dyfunc_noam_decay


class TestDygraphBasicApi_PiecewiseDecay(TestDygraphBasicApi_CosineDecay):
    def setUp(self):
        self.dygraph_func = dyfunc_piecewise_decay


class TestDygraphBasicApi_PolynomialDecay(TestDygraphBasicApi_CosineDecay):
    def setUp(self):
        self.dygraph_func = dyfunc_polynomial_decay

    def get_dygraph_output(self):
        paddle.seed(SEED)
        res = self.dygraph_func()
        return res


if __name__ == '__main__':
    unittest.main()
