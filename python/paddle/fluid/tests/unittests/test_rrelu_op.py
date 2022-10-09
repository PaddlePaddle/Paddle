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

import unittest
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.core as core
from op_test import OpTest
import paddle
import paddle.nn.functional as F
from paddle.fluid import dygraph

paddle.seed(102)
np.random.seed(102)


def ref_rrelu(x, lower, upper):
    x_t = x.copy()
    alpha = (lower + upper) / 2.0
    return np.where(x_t <= 0, alpha * x_t, x_t)


def ref_rrelu_nn(x, lower, upper):
    return ref_rrelu(x, lower, upper)


def check_output(input, output, lower, upper):
    lower_res = np.where(input <= 0, lower * input, input)
    upper_res = np.where(input <= 0, upper * input, input)
    return (output <= lower_res).all() and (output >= upper_res).all()


class TestFunctionalRReluAPI(unittest.TestCase):

    def setUp(self):
        self.x_np = np.random.uniform(-1., 1., [1, 2, 3, 4]).astype('float64')
        self.lower_0 = 0.05
        self.lower_1 = 0.1
        self.upper_0 = 0.25
        self.upper_1 = 0.33

        self.places = [
            fluid.CUDAPlace(0)
            if core.is_compiled_with_cuda() else fluid.CPUPlace()
        ]

    def check_static_result(self, place):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input = fluid.data(name="input",
                               shape=[2, 3, 4, 5],
                               dtype="float32")
            res1 = F.rrelu(x=input,
                           lower=self.lower_0,
                           upper=self.upper_0,
                           training=False)
            res2 = F.rrelu(x=input,
                           lower=self.lower_1,
                           upper=self.upper_1,
                           training=False)
            in_np = np.random.uniform(-1., 1., [2, 3, 4, 5]).astype("float32")

            res_np1 = ref_rrelu(in_np, self.lower_0, self.upper_0)
            exe = fluid.Executor(place)
            fetches = exe.run(fluid.default_main_program(),
                              feed={"input": in_np},
                              fetch_list=[res1])

            np.testing.assert_allclose(fetches[0], res_np1, rtol=1e-05)

            res_np2 = ref_rrelu(in_np, self.lower_1, self.upper_1)
            fetches = exe.run(fluid.default_main_program(),
                              feed={"input": in_np},
                              fetch_list=[res2])
            np.testing.assert_allclose(fetches[0], res_np2, rtol=1e-05)

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def test_static_graph_functional(self):
        '''test_static_graph_functional'''

        for place in self.places:
            paddle.enable_static()
            x_1 = paddle.fluid.data(name="x",
                                    shape=self.x_np.shape,
                                    dtype="float64")
            x_2 = paddle.fluid.data(name="x2",
                                    shape=self.x_np.shape,
                                    dtype="float64")
            out_1 = F.rrelu(x_1, self.lower_0, self.upper_0, training=False)
            out_2 = F.rrelu(x_2, self.lower_1, self.upper_1, training=False)
            out_3 = F.rrelu(x_2, self.lower_1, self.upper_1, training=True)

            exe = paddle.static.Executor(place=place)
            res_1, = exe.run(fluid.default_main_program(),
                             feed={"x": self.x_np},
                             fetch_list=out_1,
                             use_prune=True)
            res_2, = exe.run(fluid.default_main_program(),
                             feed={"x2": self.x_np},
                             fetch_list=out_2,
                             use_prune=True)
            res_3, = exe.run(fluid.default_main_program(),
                             feed={"x2": self.x_np},
                             fetch_list=out_3,
                             use_prune=True)

            out_ref_1 = ref_rrelu(self.x_np, self.lower_0, self.upper_0)
            out_ref_2 = ref_rrelu(self.x_np, self.lower_1, self.upper_1)
            np.testing.assert_allclose(out_ref_1, res_1, rtol=1e-05)
            np.testing.assert_allclose(out_ref_2, res_2, rtol=1e-05)
            self.assertTrue(
                check_output(self.x_np, res_3[0], self.lower_1, self.upper_1))

    def test_static_graph_layer(self):
        '''test_static_graph_layer'''

        for place in self.places:
            paddle.enable_static()
            x_1 = paddle.fluid.data(name="x",
                                    shape=self.x_np.shape,
                                    dtype="float64")
            x_2 = paddle.fluid.data(name="x2",
                                    shape=self.x_np.shape,
                                    dtype="float64")
            # init instance
            rrelu_1 = paddle.nn.RReLU(self.lower_0, self.upper_0)
            rrelu_2 = paddle.nn.RReLU(self.lower_1, self.upper_1)
            out_1 = rrelu_1(x_1)
            out_2 = rrelu_2(x_2)

            exe = paddle.static.Executor(place=place)
            res_1 = exe.run(fluid.default_main_program(),
                            feed={"x": self.x_np},
                            fetch_list=out_1,
                            use_prune=True)
            res_2 = exe.run(fluid.default_main_program(),
                            feed={"x2": self.x_np},
                            fetch_list=out_2,
                            use_prune=True)

            self.assertTrue(
                check_output(self.x_np, res_1[0], self.lower_0, self.upper_0))
            self.assertTrue(
                check_output(self.x_np, res_2[0], self.lower_1, self.upper_1))

    def dygraph_check(self, lower, upper):
        for place in self.places:
            paddle.disable_static(place)
            x = paddle.to_tensor(self.x_np)
            out = F.rrelu(x, lower, upper, training=False)
            out_ref = ref_rrelu(self.x_np, lower, upper)
            np.testing.assert_allclose(out_ref, out, rtol=1e-05)
            paddle.enable_static()

    def test_dygraph_functional(self):
        '''test_dygraph_functional'''

        self.dygraph_check(self.lower_0, self.upper_0)
        self.dygraph_check(self.lower_1, self.upper_1)

    def test_dygraph_layer(self):
        '''test_dygraph_layer'''

        for place in self.places:
            paddle.disable_static(place=place)
            rrelu = paddle.nn.RReLU(self.lower_0, self.upper_0)
            result = rrelu(paddle.to_tensor(self.x_np))
            self.assertTrue(
                check_output(self.x_np, result.numpy(), self.lower_0,
                             self.upper_0))
            paddle.enable_static()

    def test_dygraph(self):
        for place in self.places:
            paddle.disable_static(place=place)
            with dygraph.guard():
                rrelu = paddle.nn.RReLU(self.lower_0, self.upper_0)
                out_np = rrelu(paddle.to_tensor(self.x_np))
            self.assertTrue(
                check_output(self.x_np, out_np.numpy(), self.lower_0,
                             self.upper_0))
            paddle.enable_static()

    def test_error_functional(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError,
                              F.rrelu,
                              x=1,
                              lower=self.lower_0,
                              upper=self.upper_0)
            # The input dtype must be float16, float32, float64.
            x_int32 = paddle.fluid.data(name='x_int32',
                                        shape=[2, 3],
                                        dtype='int32')
            self.assertRaises(TypeError,
                              F.rrelu,
                              x=x_int32,
                              lower=self.lower_0,
                              upper=self.upper_0)
            x_bool = paddle.fluid.data(name='x_bool',
                                       shape=[2, 3],
                                       dtype='int32')
            self.assertRaises(TypeError,
                              F.rrelu,
                              x=x_bool,
                              lower=self.lower_0,
                              upper=self.upper_0)
            # lower and upper must be float
            x_fp32 = paddle.fluid.data(name='x_fp32',
                                       shape=[2, 3],
                                       dtype='float32')
            self.assertRaises(TypeError, F.rrelu, x=x_fp32, lower=0, upper=0.5)
            self.assertRaises(TypeError, F.rrelu, x=x_fp32, lower=0.5, upper=1)
            # lower and upper must be in (0, 1)
            self.assertRaises(ValueError,
                              F.rrelu,
                              x=x_fp32,
                              lower=-1.,
                              upper=0.5)
            self.assertRaises(ValueError,
                              F.rrelu,
                              x=x_fp32,
                              lower=0.5,
                              upper=2.)
            # upper should not be less than lower
            self.assertRaises(ValueError,
                              F.rrelu,
                              x=x_fp32,
                              lower=0.5,
                              upper=0.2)
            # support the input dtype is float16
            x_fp16 = paddle.fluid.data(name='x_fp16',
                                       shape=[2, 3],
                                       dtype='float16')
            F.rrelu(x=x_fp16, lower=self.lower_0, upper=self.upper_0)

    def test_error_layer(self):

        def error_int_dtype():
            with paddle.fluid.dygraph.guard():
                x = np.random.random([2, 3]).astype("float64")
                rrelu = paddle.nn.RReLU(2, 3)
                rrelu(paddle.to_tensor(x))

        def error_lower_dtype():
            with paddle.fluid.dygraph.guard():
                x = np.random.random([2, 3]).astype("float32")
                rrelu = paddle.nn.RReLU(0, 0.5)
                rrelu(paddle.to_tensor(x))

        def error_upper_dtype():
            with paddle.fluid.dygraph.guard():
                x = np.random.random([2, 3]).astype("float32")
                rrelu = paddle.nn.RReLU(0.5, 1)
                rrelu(paddle.to_tensor(x))

        def error_lower_range():
            with paddle.fluid.dygraph.guard():
                x = np.random.random([2, 3]).astype("float32")
                rrelu = paddle.nn.RReLU(-1.0, 0.5)
                rrelu(paddle.to_tensor(x))

        def error_upper_range():
            with paddle.fluid.dygraph.guard():
                x = np.random.random([2, 3]).astype("float32")
                rrelu = paddle.nn.RReLU(0.5, 2.0)
                rrelu(paddle.to_tensor(x))

        def error_lower_upper():
            with paddle.fluid.dygraph.guard():
                x = np.random.random([2, 3]).astype("float32")
                rrelu = paddle.nn.RReLU(0.5, 0.2)
                rrelu(paddle.to_tensor(x))

        self.assertRaises(TypeError, error_int_dtype)
        self.assertRaises(TypeError, error_lower_dtype)
        self.assertRaises(TypeError, error_upper_dtype)
        self.assertRaises(ValueError, error_lower_range)
        self.assertRaises(ValueError, error_upper_range)
        self.assertRaises(ValueError, error_lower_upper)


class RReluTest(OpTest):

    def setUp(self):
        self.op_type = "rrelu"
        self.lower = 0.1
        self.upper = 0.3
        self.is_test = True
        self.init_prams()

    def init_prams(self):
        self.dtype = "float64"
        self.x_shape = [2, 3, 4, 5]

        x_np = np.random.uniform(-1, 1, self.x_shape).astype(self.dtype)
        out_np = ref_rrelu(x_np, self.lower, self.upper)
        noise_np = np.ones(self.x_shape).astype(self.dtype)
        noise_np[x_np < 0] = (self.lower + self.upper) / 2.0

        self.inputs = {'X': x_np}
        self.outputs = {'Out': out_np, 'Noise': noise_np}
        self.attrs = {
            'lower': self.lower,
            "upper": self.upper,
            "is_test": self.is_test
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class RReluTrainingTest(OpTest):

    def setUp(self):
        self.op_type = "rrelu"
        self.lower = 0.3
        self.upper = 0.3000009
        self.is_test = False
        self.init_prams()


class RReluTrainingTest(OpTest):

    def setUp(self):
        self.op_type = "rrelu"
        self.lower = 0.3
        self.upper = 0.3000009
        self.is_test = False
        self.init_prams()


if __name__ == "__main__":
    unittest.main()
