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

import inspect
import unittest

import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
from paddle import to_tensor
from paddle.fluid.dygraph import to_variable
from paddle.jit.api import dygraph_to_static_func
from paddle.jit.dy2static.utils import is_dygraph_api
from paddle.utils import gast

SEED = 2020
np.random.seed(SEED)

# TODO(zhhsplendid): This test is old so that use a static graph style
# mark it as TODO, to refactoring the code of this file.
paddle.enable_static()


def dyfunc_to_variable(x):
    res = fluid.dygraph.to_variable(x, name=None, zero_copy=None)
    return res


def dyfunc_to_variable_2(x):
    res = dygraph.to_variable(value=np.zeros(shape=(1), dtype=np.int32))
    return res


def dyfunc_to_variable_3(x):
    res = to_variable(x, name=None, zero_copy=None)
    return res


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


class TestDygraphBasicApi_ToVariable(unittest.TestCase):
    def setUp(self):
        self.input = np.ones(5).astype("int32")
        self.test_funcs = [
            dyfunc_to_tensor,
            dyfunc_bool_to_tensor,
            dyfunc_int_to_tensor,
            dyfunc_float_to_tensor,
            dyfunc_to_variable,
            dyfunc_to_variable_2,
            dyfunc_to_variable_3,
        ]
        self.place = (
            fluid.CUDAPlace(0)
            if fluid.is_compiled_with_cuda()
            else fluid.CPUPlace()
        )

    def get_dygraph_output(self):
        with fluid.dygraph.guard():
            res = self.dygraph_func(self.input).numpy()
            return res

    def get_static_output(self):
        main_program = fluid.Program()
        main_program.random_seed = SEED
        with fluid.program_guard(main_program):
            static_out = dygraph_to_static_func(self.dygraph_func)(self.input)

        exe = fluid.Executor(self.place)
        static_res = exe.run(main_program, fetch_list=static_out)

        return static_res[0]

    def test_transformed_static_result(self):
        for func in self.test_funcs:
            self.dygraph_func = func
            dygraph_res = self.get_dygraph_output()
            static_res = self.get_static_output()
            np.testing.assert_allclose(dygraph_res, static_res, rtol=1e-05)


# 1. test Apis that inherit from layers.Layer
def dyfunc_BilinearTensorProduct(layer1, layer2):
    bilinearTensorProduct = paddle.nn.Bilinear(
        5,
        4,
        1000,
        weight_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.99)
        ),
        bias_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.5)
        ),
    )

    res = bilinearTensorProduct(
        fluid.dygraph.base.to_variable(layer1),
        fluid.dygraph.base.to_variable(layer2),
    )
    return res


def dyfunc_Conv2D(input):
    conv2d = paddle.nn.Conv2D(
        in_channels=3,
        out_channels=2,
        kernel_size=3,
        weight_attr=paddle.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.99)
        ),
        bias_attr=paddle.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.5)
        ),
    )
    res = conv2d(input)
    return res


def dyfunc_Conv3D(input):
    conv3d = paddle.nn.Conv3D(
        in_channels=3,
        out_channels=2,
        kernel_size=3,
        weight_attr=paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(value=0.99)
        ),
        bias_attr=fluid.ParamAttr(
            initializer=paddle.nn.initializer.Constant(value=0.5)
        ),
    )
    res = conv3d(input)
    return res


def dyfunc_Conv2DTranspose(input):
    conv2dTranspose = paddle.nn.Conv2DTranspose(
        3,
        12,
        12,
        weight_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.99)
        ),
        bias_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.5)
        ),
    )
    ret = conv2dTranspose(input)
    return ret


def dyfunc_Conv3DTranspose(input):
    conv3dTranspose = paddle.nn.Conv3DTranspose(
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
    ret = conv3dTranspose(input)
    return ret


def dyfunc_Linear(input):
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
    res = fc(input)
    return m(res)


def dyfunc_Pool2D(input):
    paddle.nn.AvgPool2D(kernel_size=2, stride=1)
    pool2d = paddle.nn.AvgPool2D(kernel_size=2, stride=1)
    res = pool2d(input)
    return res


def dyfunc_Prelu(input):
    prelu0 = paddle.nn.PReLU(
        weight_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(1.0)
        ),
    )
    res = prelu0(input)
    return res


class TestDygraphBasicApi(unittest.TestCase):
    # Compare results of dynamic graph and transformed static graph function which only
    # includes basic Api.

    def setUp(self):
        self.input = np.random.random((1, 4, 3, 3)).astype('float32')
        self.dygraph_func = dyfunc_Pool2D

    def get_dygraph_output(self):
        with fluid.dygraph.guard():
            fluid.default_startup_program.random_seed = SEED
            fluid.default_main_program.random_seed = SEED
            data = fluid.dygraph.to_variable(self.input)
            res = self.dygraph_func(data).numpy()

            return res

    def get_static_output(self):
        startup_program = fluid.Program()
        startup_program.random_seed = SEED
        main_program = fluid.Program()
        main_program.random_seed = SEED
        with fluid.program_guard(main_program, startup_program):
            data = fluid.layers.assign(self.input)
            static_out = dygraph_to_static_func(self.dygraph_func)(data)

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(startup_program)
        static_res = exe.run(main_program, fetch_list=static_out)
        return static_res[0]

    def test_transformed_static_result(self):
        dygraph_res = self.get_dygraph_output()
        static_res = self.get_static_output()
        np.testing.assert_allclose(dygraph_res, static_res, rtol=1e-05)


class TestDygraphBasicApi_BilinearTensorProduct(TestDygraphBasicApi):
    def setUp(self):
        self.input1 = np.random.random((5, 5)).astype('float32')
        self.input2 = np.random.random((5, 4)).astype('float32')
        self.dygraph_func = dyfunc_BilinearTensorProduct

    def get_dygraph_output(self):
        with fluid.dygraph.guard():
            fluid.default_startup_program.random_seed = SEED
            fluid.default_main_program.random_seed = SEED
            res = self.dygraph_func(self.input1, self.input2).numpy()
            return res

    def get_static_output(self):
        startup_program = fluid.Program()
        startup_program.random_seed = SEED
        main_program = fluid.Program()
        main_program.random_seed = SEED
        with fluid.program_guard(main_program, startup_program):
            static_out = dygraph_to_static_func(self.dygraph_func)(
                self.input1, self.input2
            )

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(startup_program)
        static_res = exe.run(main_program, fetch_list=static_out)
        return static_res[0]


class TestDygraphBasicApi_Conv2D(TestDygraphBasicApi):
    def setUp(self):
        self.input = np.random.random((1, 3, 3, 5)).astype('float32')
        self.dygraph_func = dyfunc_Conv2D


class TestDygraphBasicApi_Conv3D(TestDygraphBasicApi):
    def setUp(self):
        self.input = np.random.random((1, 3, 3, 3, 5)).astype('float32')
        self.dygraph_func = dyfunc_Conv3D


class TestDygraphBasicApi_Conv2DTranspose(TestDygraphBasicApi):
    def setUp(self):
        self.input = np.random.random((5, 3, 32, 32)).astype('float32')
        self.dygraph_func = dyfunc_Conv2DTranspose


class TestDygraphBasicApi_Conv3DTranspose(TestDygraphBasicApi):
    def setUp(self):
        self.input = np.random.random((5, 3, 12, 32, 32)).astype('float32')
        self.dygraph_func = dyfunc_Conv3DTranspose


class TestDygraphBasicApi_Linear(TestDygraphBasicApi):
    def setUp(self):
        self.input = np.random.random((4, 3, 10)).astype('float32')
        self.dygraph_func = dyfunc_Linear


class TestDygraphBasicApi_Prelu(TestDygraphBasicApi):
    def setUp(self):
        self.input = np.ones([5, 20, 10, 10]).astype('float32')
        self.dygraph_func = dyfunc_Prelu


# 2. test Apis that inherit from LearningRateDecay
def dyfunc_CosineDecay():
    base_lr = 0.1
    CosineDecay = fluid.dygraph.CosineDecay(
        learning_rate=base_lr, step_each_epoch=10000, epochs=120
    )
    lr = CosineDecay()
    return lr


def dyfunc_ExponentialDecay():
    base_lr = 0.1
    exponential_decay = fluid.dygraph.ExponentialDecay(
        learning_rate=base_lr, decay_steps=10000, decay_rate=0.5, staircase=True
    )
    lr = exponential_decay()
    return lr


def dyfunc_InverseTimeDecay():
    base_lr = 0.1
    inverse_time_decay = fluid.dygraph.InverseTimeDecay(
        learning_rate=base_lr, decay_steps=10000, decay_rate=0.5, staircase=True
    )
    lr = inverse_time_decay()
    return lr


def dyfunc_NaturalExpDecay():
    base_lr = 0.1
    natural_exp_decay = fluid.dygraph.NaturalExpDecay(
        learning_rate=base_lr, decay_steps=10000, decay_rate=0.5, staircase=True
    )
    lr = natural_exp_decay()
    return lr


def dyfunc_NoamDecay():
    noam_decay = fluid.dygraph.NoamDecay(100, 100)
    lr = noam_decay()
    return lr


def dyfunc_PiecewiseDecay():
    boundaries = [10000, 20000]
    values = [1.0, 0.5, 0.1]
    pd = fluid.dygraph.PiecewiseDecay(boundaries, values, begin=0)
    lr = pd()
    return lr


def dyfunc_PolynomialDecay():
    start_lr = 0.01
    total_step = 5000
    end_lr = 0
    pd = fluid.dygraph.PolynomialDecay(start_lr, total_step, end_lr, power=1.0)
    lr = pd()
    return lr


class TestDygraphBasicApi_CosineDecay(unittest.TestCase):
    def setUp(self):
        self.dygraph_func = dyfunc_CosineDecay

    def get_dygraph_output(self):
        with fluid.dygraph.guard():
            fluid.default_startup_program.random_seed = SEED
            fluid.default_main_program.random_seed = SEED
            res = self.dygraph_func().numpy()
            return res

    def get_static_output(self):
        startup_program = fluid.Program()
        startup_program.random_seed = SEED
        main_program = fluid.Program()
        main_program.random_seed = SEED
        with fluid.program_guard(main_program, startup_program):
            static_out = dygraph_to_static_func(self.dygraph_func)()

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(startup_program)
        static_res = exe.run(main_program, fetch_list=static_out)
        return static_res[0]

    def test_transformed_static_result(self):
        dygraph_res = self.get_dygraph_output()
        static_res = self.get_static_output()
        np.testing.assert_allclose(dygraph_res, static_res, rtol=1e-05)


class TestDygraphBasicApi_ExponentialDecay(TestDygraphBasicApi_CosineDecay):
    def setUp(self):
        self.dygraph_func = dyfunc_ExponentialDecay


class TestDygraphBasicApi_InverseTimeDecay(TestDygraphBasicApi_CosineDecay):
    def setUp(self):
        self.dygraph_func = dyfunc_InverseTimeDecay


class TestDygraphBasicApi_NaturalExpDecay(TestDygraphBasicApi_CosineDecay):
    def setUp(self):
        self.dygraph_func = dyfunc_NaturalExpDecay


class TestDygraphBasicApi_NoamDecay(TestDygraphBasicApi_CosineDecay):
    def setUp(self):
        self.dygraph_func = dyfunc_NoamDecay


class TestDygraphBasicApi_PiecewiseDecay(TestDygraphBasicApi_CosineDecay):
    def setUp(self):
        self.dygraph_func = dyfunc_PiecewiseDecay


class TestDygraphBasicApi_PolynomialDecay(TestDygraphBasicApi_CosineDecay):
    def setUp(self):
        self.dygraph_func = dyfunc_PolynomialDecay


def _dygraph_fn():
    import paddle.fluid as fluid

    x = np.random.random((1, 3)).astype('float32')
    with fluid.dygraph.guard():
        fluid.dygraph.to_variable(x)
        np.random.random((1))


class TestDygraphApiRecognition(unittest.TestCase):
    def setUp(self):
        self.src = inspect.getsource(_dygraph_fn)
        self.root = gast.parse(self.src)

    def _get_dygraph_ast_node(self):
        return self.root.body[0].body[2].body[0].value

    def _get_static_ast_node(self):
        return self.root.body[0].body[2].body[1].value

    def test_dygraph_api(self):
        self.assertTrue(is_dygraph_api(self._get_dygraph_ast_node()) is True)
        self.assertTrue(is_dygraph_api(self._get_static_ast_node()) is False)


if __name__ == '__main__':
    unittest.main()
