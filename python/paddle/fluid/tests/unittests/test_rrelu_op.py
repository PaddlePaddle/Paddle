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

import unittest
import numpy as np
import paddle.fluid as fluid
import six
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard
from op_test import OpTest
import paddle
import paddle.nn.functional as F

import time
def debug_log(msg,is_clear=False):
    fp = open('/tmp/data.txt', 'w' if is_clear else "a")
    fp.write(str(time.time()) + " => " + msg + "\n")
    fp.close()

debug_log("=======> 111", True)

xx= paddle.rand((2, 3))
rrelu1 = paddle.nn.RReLU()
print(rrelu1(xx))
print(F.rrelu(xx, 0.1, 0.4, training = True))

def ref_rrelu(x, lower, upper):
    x_t = x.copy()
    alpha = (lower + upper) / 2.0
    return np.where(x_t <= 0, alpha * x_t, x_t)

def ref_rrelu_nn(x, lower, upper):
    return ref_rrelu(x, lower, upper)

def check_output(input, output, lower, upper):
    lower_res = np.where(input <= 0, lower * input, input)
    upper_res = np.where(input <= 0, upper * input, input)
    return (output >= lower_res).all() and (output <= upper_res).all()

class TestFunctionalRReluAPI(unittest.TestCase):
    def setUp(self):
        # self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda(
        # ) else paddle.CPUPlace()
        self.place = paddle.CPUPlace()
        self.x_np = np.random.uniform(-1., 1., [1, 2, 3, 4]).astype('float32')
        self.lower_0 = 0.05
        self.lower_1 = 0.1
        self.upper_0 = 0.25
        self.upper_1 = 0.33
        debug_log("=======> 222")

    def static_check(self, lower, upper):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', self.x_np.shape, 'float32')
            out = F.rrelu(x, lower, upper)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np},
                          fetch_list=[out])
        out_ref = ref_rrelu(self.x_np, lower, upper)
        self.assertEqual(np.allclose(out_ref, res[0]), True)

    def dygraph_check(self, lower, upper):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out = F.rrelu(x, lower, upper)
        out_ref = ref_rrelu(self.x_np, lower, upper)
        self.assertEqual(np.allclose(out_ref, out.numpy()), True)
        paddle.enable_static()

    def test_static_api(self):
        self.static_check(self.lower_0, self.upper_0)
        self.static_check(self.lower_1, self.upper_1)

    # def test_dygraph_api(self):
    #     self.dygraph_check(self.lower_0, self.upper_0)
    #     self.dygraph_check(self.lower_1, self.upper_1)

    def test_error_functional(self):
        with paddle.static.program_guard(paddle.static.Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, F.rrelu, x=1, lower=self.lower_0, upper=self.upper_0)
            # The input dtype must be float16, float32, float64.
            x_int32 = paddle.fluid.data(
                name='x_int32', shape=[2, 3], dtype='int32')
            self.assertRaises(TypeError, F.rrelu, x=x_int32, lower=self.lower_0, upper=self.upper_0)
            x_bool = paddle.fluid.data(
                name='x_bool', shape=[2, 3], dtype='int32')
            self.assertRaises(TypeError, F.rrelu, x=x_bool, lower=self.lower_0, upper=self.upper_0)
            # lower and upper must be float
            x_fp32 = paddle.fluid.data(
                name='x_fp32', shape=[2, 3], dtype='float32')
            self.assertRaises(TypeError, F.rrelu, x=x_fp32, lower=0, upper=0.5)
            self.assertRaises(TypeError, F.rrelu, x=x_fp32, lower=0.5, upper=1)
            # lower and upper must be in (0, 1)
            self.assertRaises(ValueError, F.rrelu, x=x_fp32, lower=-1., upper=0.5)
            self.assertRaises(ValueError, F.rrelu, x=x_fp32, lower=0.5, upper=2.)
            # upper should not be less than lower
            self.assertRaises(ValueError, F.rrelu, x=x_fp32, lower=0.5, upper=0.2)
            # support the input dtype is float16
            x_fp16 = paddle.fluid.data(
                name='x_fp16', shape=[2, 3], dtype='float16')
            F.rrelu(x=x_fp16, lower=self.lower_0, upper=self.upper_0)

    def test_error_layer(self):
        def error_variable():
            # The input type must be Variable.
            with paddle.fluid.dygraph.guard():
                x = 6
                rrelu = paddle.nn.RReLU(self.lower_0, self.upper_0)
                rrelu(paddle.to_tensor(x))

        def error_int_dtype():
            with paddle.fluid.dygraph.guard():
                x = np.random.random([2, 3]).astype("int32")
                rrelu = paddle.nn.RReLU(self.lower_0, self.upper_0)
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

        self.assertRaises(TypeError, error_variable)
        # self.assertRaises(TypeError, error_int_dtype)
        # self.assertRaises(TypeError, error_lower_dtype)
        # self.assertRaises(TypeError, error_upper_dtype)
        # self.assertRaises(ValueError, error_lower_range)
        # self.assertRaises(ValueError, error_upper_range)
        # self.assertRaises(ValueError, error_lower_upper)


# class TestRReluAPI(unittest.TestCase):
#     def setUp(self):
#         self.shape = [2, 3]
#         self.x_1_np = np.random.random(self.shape).astype("float64")
#         self.lower = 0.1
#         self.upper = 0.25
#         debug_log("=======> 333")
#
#     def test_static_graph_functional(self):
#         for use_cuda in ([False, True]
#                          if core.is_compiled_with_cuda() else [False]):
#             place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
#             paddle.enable_static()
#             x_1 = paddle.fluid.data(
#                 name="X", shape=self.shape, dtype="float64")
#             out_1 = F.rrelu(x_1, self.lower, self.upper)
#             exe = paddle.static.Executor(place=place)
#             res_1 = exe.run(fluid.default_main_program(),
#                             feed={"X": self.x_1_np},
#                             fetch_list=out_1,
#                             use_prune=True)
#             self.assertTrue(check_output(self.x_1_np, res_1, self.lower, self.upper))
#
#     # same test between layer and functional in this op.
#     def test_static_graph_layer(self):
#         for use_cuda in ([False, True]
#                          if core.is_compiled_with_cuda() else [False]):
#             place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
#
#             paddle.enable_static()
#             x_1 = paddle.fluid.data(
#                 name="X", shape=self.shape, dtype="float64")
#
#             # init instance
#             ps_1 = paddle.nn.RReLU(self.lower, self.upper)
#             out_1 = ps_1(x_1)
#             exe = paddle.static.Executor(place=place)
#             res_1 = exe.run(fluid.default_main_program(),
#                             feed={"X": self.x_1_np},
#                             fetch_list=out_1,
#                             use_prune=True)
#             self.assertTrue(check_output(self.x_1_np, res_1, self.lower, self.upper))
#
#     def test_dygraph(self):
#         for use_cuda in ([False, True]
#                          if core.is_compiled_with_cuda() else [False]):
#             place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
#
#             paddle.disable_static(place=place)
#
#             rrelu = paddle.nn.RReLU(self.lower, self.upper)
#             result = rrelu(paddle.to_tensor(self.x_1_np))
#             self.assertTrue(check_output(self.x_1_np, result.numpy(), self.lower, self.upper))
#             result_functional = F.rrelu(
#                 paddle.to_tensor(self.x_1_np), self.lower, self.upper)
#             self.assertTrue(check_output(self.x_1_np, result_functional.numpy(), self.lower, self.upper))


# class RReluTest(OpTest):
#     def setUp(self):
#         self.init_alpha()
#         self.init_dtype()
#         self.init_input_shape()
#         self.init_attr()
#         self.op_type = "rrelu"
#
#         x_np = np.random.uniform(-1, 1, self.x_shape).astype(self.dtype)
#         x_np[np.abs(x_np) < 0.005] = 0.02
#         out_np = ref_rrelu(x_np, self.alpha, self.alpha)
#         self.inputs = {'X': x_np}
#         self.outputs = {'Out': out_np}
#         debug_log("=======> 444")
#
#     def init_alpha(self):
#         self.alpha = 0.5
#
#     def init_dtype(self):
#         self.dtype = np.float64
#
#     def init_input_shape(self):
#         self.x_shape = [2, 3]
#
#     def init_attr(self):
#         self.attrs = {'lower': self.alpha, "upper": self.alpha}
#
#     def test_check_output(self):
#         self.check_output()
#
#     def test_check_grad(self):
#         self.check_grad(['X'], 'Out')

if __name__ == "__main__":
    unittest.main()
