# #   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# from __future__ import print_function

# import unittest
# import numpy as np
# import paddle.fluid as fluid
# import paddle.fluid.core as core
# from op_test import OpTest
# import paddle
# import paddle.nn.functional as F
# from paddle.fluid import dygraph

# paddle.seed(102)
# np.random.seed(102)


# def ref_rrelu(x, lower, upper):
#     x_t = x.copy()
#     alpha = (lower + upper) / 2.0
#     return np.where(x_t <= 0, alpha * x_t, x_t)


# def ref_rrelu_nn(x, lower, upper):
#     return ref_rrelu(x, lower, upper)


# def check_output(input, output, lower, upper):
#     lower_res = np.where(input <= 0, lower * input, input)
#     upper_res = np.where(input <= 0, upper * input, input)
#     return (output <= lower_res).all() and (output >= upper_res).all()


# class TestFunctionalRReluAPI(unittest.TestCase):
#     def setUp(self):
#         self.x_np = np.random.uniform(-1., 1., [1, 2, 3, 4]).astype('float64')
#         self.lower_0 = 0.05
#         self.lower_1 = 0.1
#         self.upper_0 = 0.25
#         self.upper_1 = 0.33

#         self.places = [
#             fluid.CUDAPlace(0)
#             if core.is_compiled_with_cuda() else fluid.CPUPlace()
#         ]

#     def check_static_result(self, place):
#         with fluid.program_guard(fluid.Program(), fluid.Program()):
#             input = fluid.data(
#                 name="input", shape=[2, 3, 4, 5], dtype="float32")
#             res1 = F.rrelu(
#                 x=input, lower=self.lower_0, upper=self.upper_0, training=False)
#             res2 = F.rrelu(
#                 x=input, lower=self.lower_1, upper=self.upper_1, training=False)
#             in_np = np.random.uniform(-1., 1., [2, 3, 4, 5]).astype("float32")

#             res_np1 = ref_rrelu(in_np, self.lower_0, self.upper_0)
#             exe = fluid.Executor(place)
#             fetches = exe.run(fluid.default_main_program(),
#                               feed={"input": in_np},
#                               fetch_list=[res1])

#             self.assertTrue(np.allclose(fetches[0], res_np1))

#             res_np2 = ref_rrelu(in_np, self.lower_1, self.upper_1)
#             fetches = exe.run(fluid.default_main_program(),
#                               feed={"input": in_np},
#                               fetch_list=[res2])
#             self.assertTrue(np.allclose(fetches[0], res_np2))

#     def test_static(self):
#         for place in self.places:
#             self.check_static_result(place=place)

#     def test_static_graph_functional(self):
#         '''test_static_graph_functional'''

#         for place in self.places:
#             paddle.enable_static()
#             x_1 = paddle.fluid.data(
#                 name="x", shape=self.x_np.shape, dtype="float64")
#             x_2 = paddle.fluid.data(
#                 name="x2", shape=self.x_np.shape, dtype="float64")
#             out_1 = F.rrelu(x_1, self.lower_0, self.upper_0, training=False)
#             out_2 = F.rrelu(x_2, self.lower_1, self.upper_1, training=False)
#             out_3 = F.rrelu(x_2, self.lower_1, self.upper_1, training=True)

#             exe = paddle.static.Executor(place=place)
#             res_1 = exe.run(fluid.default_main_program(),
#                             feed={"x": self.x_np},
#                             fetch_list=out_1,
#                             use_prune=True)
#             res_2 = exe.run(fluid.default_main_program(),
#                             feed={"x2": self.x_np},
#                             fetch_list=out_2,
#                             use_prune=True)
#             res_3 = exe.run(fluid.default_main_program(),
#                             feed={"x2": self.x_np},
#                             fetch_list=out_3,
#                             use_prune=True)

#             out_ref_1 = ref_rrelu(self.x_np, self.lower_0, self.upper_0)
#             out_ref_2 = ref_rrelu(self.x_np, self.lower_1, self.upper_1)
#             self.assertEqual(np.allclose(out_ref_1, res_1), True)
#             self.assertEqual(np.allclose(out_ref_2, res_2), True)
#             self.assertTrue(
#                 check_output(self.x_np, res_3[0], self.lower_1, self.upper_1))
                                     
#     def test_static_graph_layer(self):
#         '''test_static_graph_layer'''

#         for place in self.places:
#             paddle.enable_static()
#             x_1 = paddle.fluid.data(
#                 name="x", shape=self.x_np.shape, dtype="float64")
#             x_2 = paddle.fluid.data(
#                 name="x2", shape=self.x_np.shape, dtype="float64")
#             # init instance
#             rrelu_1 = paddle.nn.RReLU(self.lower_0, self.upper_0)
#             rrelu_2 = paddle.nn.RReLU(self.lower_1, self.upper_1)
#             out_1 = rrelu_1(x_1)
#             out_2 = rrelu_2(x_2)

#             exe = paddle.static.Executor(place=place)
#             res_1 = exe.run(fluid.default_main_program(),
#                             feed={"x": self.x_np},
#                             fetch_list=out_1,
#                             use_prune=True)
#             res_2 = exe.run(fluid.default_main_program(),
#                             feed={"x2": self.x_np},
#                             fetch_list=out_2,
#                             use_prune=True)

#             self.assertTrue(
#                 check_output(self.x_np, res_1[0], self.lower_0, self.upper_0))
#             self.assertTrue(
#                 check_output(self.x_np, res_2[0], self.lower_1, self.upper_1))

#     def dygraph_check(self, lower, upper):
#         for place in self.places:
#             paddle.disable_static(place)
#             x = paddle.to_tensor(self.x_np)
#             out = F.rrelu(x, lower, upper, training=False)
#             out_ref = ref_rrelu(self.x_np, lower, upper)
#             self.assertEqual(np.allclose(out_ref, out), True)
#             paddle.enable_static()

#     def test_dygraph_functional(self):
#         '''test_dygraph_functional'''

#         self.dygraph_check(self.lower_0, self.upper_0)
#         self.dygraph_check(self.lower_1, self.upper_1)

#     def test_dygraph_layer(self):
#         '''test_dygraph_layer'''

#         for place in self.places:
#             paddle.disable_static(place=place)
#             rrelu = paddle.nn.RReLU(self.lower_0, self.upper_0)
#             result = rrelu(paddle.to_tensor(self.x_np))
#             self.assertTrue(
#                 check_output(self.x_np,
#                              result.numpy(), self.lower_0, self.upper_0))
#             paddle.enable_static()

#     def test_dygraph(self):
#         for place in self.places:
#             paddle.disable_static(place=place)
#             with dygraph.guard():
#                 rrelu = paddle.nn.RReLU(self.lower_0, self.upper_0)
#                 out_np = rrelu(paddle.to_tensor(self.x_np))
#             self.assertTrue(
#                 check_output(self.x_np,
#                              out_np.numpy(), self.lower_0, self.upper_0))
#             paddle.enable_static()

#     def test_error_functional(self):
#         with paddle.static.program_guard(paddle.static.Program()):
#             # The input type must be Variable.
#             self.assertRaises(
#                 TypeError, F.rrelu, x=1, lower=self.lower_0, upper=self.upper_0)
#             # The input dtype must be float16, float32, float64.
#             x_int32 = paddle.fluid.data(
#                 name='x_int32', shape=[2, 3], dtype='int32')
#             self.assertRaises(
#                 TypeError,
#                 F.rrelu,
#                 x=x_int32,
#                 lower=self.lower_0,
#                 upper=self.upper_0)
#             x_bool = paddle.fluid.data(
#                 name='x_bool', shape=[2, 3], dtype='bool')
#             self.assertRaises(
#                 TypeError,
#                 F.rrelu,
#                 x=x_bool,
#                 lower=self.lower_0,
#                 upper=self.upper_0)
#             x_fp32 = paddle.fluid.data(
#                 name='x_fp32', shape=[2, 3], dtype='float32')
#             # lower and upper must be in [0, 1]
#             self.assertRaises(
#                 ValueError, F.rrelu, x=x_fp32, lower=-1., upper=0.5)
#             self.assertRaises(
#                 ValueError, F.rrelu, x=x_fp32, lower=0.5, upper=2.)
#             # upper should not be less than lower
#             self.assertRaises(
#                 ValueError, F.rrelu, x=x_fp32, lower=0.5, upper=0.2)
#             # support the input dtype is float16
#             x_fp16 = paddle.fluid.data(
#                 name='x_fp16', shape=[2, 3], dtype='float16')
#             F.rrelu(x=x_fp16, lower=self.lower_0, upper=self.upper_0)

#     def test_error_layer(self):
#         def error_lower_range():
#             with paddle.fluid.dygraph.guard():
#                 x = np.random.random([2, 3]).astype("float32")
#                 rrelu = paddle.nn.RReLU(-1.0, 0.5)
#                 rrelu(paddle.to_tensor(x))

#         def error_upper_range():
#             with paddle.fluid.dygraph.guard():
#                 x = np.random.random([2, 3]).astype("float32")
#                 rrelu = paddle.nn.RReLU(0.5, 2.0)
#                 rrelu(paddle.to_tensor(x))

#         def error_lower_upper():
#             with paddle.fluid.dygraph.guard():
#                 x = np.random.random([2, 3]).astype("float32")
#                 rrelu = paddle.nn.RReLU(0.5, 0.2)
#                 rrelu(paddle.to_tensor(x))

#         self.assertRaises(ValueError, error_lower_range)
#         self.assertRaises(ValueError, error_upper_range)
#         self.assertRaises(ValueError, error_lower_upper)


# class RReluTest(OpTest):
#     def setUp(self):
#         self.op_type = "rrelu"
#         self.lower = 0.1
#         self.upper = 0.3
#         self.is_test = True
#         self.init_prams()

#     def init_prams(self):
#         self.dtype = "float64"
#         self.x_shape = [2, 3, 4, 5]

#         x_np = np.random.uniform(-1, 1, self.x_shape).astype(self.dtype)
#         out_np = ref_rrelu(x_np, self.lower, self.upper)
#         mask_np = np.ones(self.x_shape).astype(self.dtype)
#         mask_np[x_np < 0] = (self.lower + self.upper) / 2.0

#         self.inputs = {'X': x_np}
#         self.outputs = {'Out': out_np, 'Mask': mask_np}
#         self.attrs = {
#             'lower': self.lower,
#             "upper": self.upper,
#             "is_test": self.is_test
#         }

#     def test_check_output(self):
#         self.check_output()

#     def test_check_grad(self):
#         self.check_grad(['X'], 'Out')


# class RReluTrainingTest(OpTest):
#     def setUp(self):
#         self.op_type = "rrelu"
#         self.lower = 0.3
#         self.upper = 0.3000009
#         self.is_test = False
#         self.init_prams()


# class RReluTrainingTest(OpTest):
#     def setUp(self):
#         self.op_type = "rrelu"
#         self.lower = 0.3
#         self.upper = 0.3000009
#         self.is_test = False
#         self.init_prams()


# if __name__ == "__main__":
#     unittest.main()









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
import paddle.fluid.core as core
from op_test import OpTest, skip_check_grad_ci, convert_float_to_uint16
import paddle
import paddle.static as static
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
from paddle.fluid.framework import _test_eager_guard, _enable_legacy_dygraph
import os

from paddle import _C_ops


def rrelu_inference(x, lower, upper):
    x_t = x.copy()
    alpha = (lower + upper) / 2.0
    return np.where(x_t < 0, alpha * x_t, x_t)


def check_element_range_of_rrelu_output_in_training(input: np.ndarray, op_output: np.ndarray, lower: float, upper: float):
    """
    input: x
    op outpout: y
    check that:
    if x[i] >= 0, then x[i] == y[i]
    if x[i] < 0, then  upper * x[i] <= y[i] <= lower * x[i]

    return: True: the test is passed;
            False: the test is not passed
    """
    passed_1 = np.allclose(input[input >= 0], op_output[input >= 0])
    if passed_1 == False:
        return False
    passed_2 = (op_output[input < 0] <= (input[input < 0] * lower)).all()
    if passed_2 == False:
        return False
    passed_3 = (op_output[input < 0] >= (input[input < 0] * upper)).all()
    return passed_3


def check_negative_elements_distribution_of_rrelu_output_in_training(input: np.ndarray, op_output: np.ndarray, lower: float, upper: float, num_segments: int, scale: float):
    """
    input: x
    op_output: y

    Only check negative elements.
    Divide the interval [lower, upper] into num_segments equal parts.
    [a, b] is a small interval in [lower, upper].  0 <=a <= b <= 1
    count = the number of i that satisfies x[i] < 0 and b * x[i] <= y[i] <= a * x[i]
    Then count / x.size >=  scale * (b - a) / (upper - lower)

    scale is recommended to be in the range [0.1, 0.8]
    
    if this check is passed, you can "roughly" believe that the function of 
    RReLU API has been properly implemented.
    the function of RReLU API can be shown as follows:
    out = np.where(x < 0, 
            np.random.uniform(lower, upper, x.shape) * x, x)
    """
    num_negative_elements = np.sum(input < 0)   
    one_part_length = (upper - lower) / num_segments
    special_alphas = []
    for i in range(num_segments):
        alpha = lower + i * one_part_length
        special_alphas.append(alpha)
    special_alphas.append(upper)

    for i in range(num_segments):
        bool_array_1 = op_output[input < 0] <= (input[input < 0] * special_alphas[i])
        bool_array_2 = op_output[input < 0] >= (input[input < 0] * special_alphas[i+1])
        count = np.sum(bool_array_1 * bool_array_2)
        # print(i, "{}%".format(count / num_negative_elements * 100))
        if count / num_negative_elements < scale * 1 / num_segments:
            return False
    return True


class TestRReluOpInference(OpTest):
    """
    test the inference mode of rrelu op,
    you can subclass this class and modify "setUp" method
    as you want
    """
    def setUp(self):
        self.op_type = "rrelu"
        self.lower = 0.1
        self.upper = 0.3
        self.fix_seed = True
        self.seed = 1
        self.dtype = "float64"
        self.x_shape = [2, 3, 4, 5]
        self.x_low = -1
        self.x_high = 1
        self.init()

    def init(self):
        x_np = np.random.uniform(self.x_low, self.x_high, self.x_shape).astype(self.dtype)
        out_np = rrelu_inference(x_np, self.lower, self.upper)
        mask_np = np.ones(self.x_shape).astype(self.dtype)
        mask_np[x_np < 0] = (self.lower + self.upper) / 2.0

        self.inputs = {'X': x_np}
        self.outputs = {'Out': out_np, 'Mask': mask_np}
        self.attrs = {
            'lower': self.lower,
            "upper": self.upper,
            "is_test": True,
            "fix_seed": self.fix_seed,
            "seed": self.seed
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestRReluOpInference2(TestRReluOpInference):
    def setUp(self):
        self.op_type = "rrelu"
        self.lower = 0.3
        self.upper = 0.99
        self.fix_seed = True
        self.seed = 198
        self.dtype = "float64"
        self.x_shape = [20, 10]
        self.x_low = -9
        self.x_high = -1
        self.init()


class TestRReluOpInference3(TestRReluOpInference):
    def setUp(self):
        self.op_type = "rrelu"
        self.lower = 0.8
        self.upper = 0.99
        self.fix_seed = False
        self.seed = 198
        self.dtype = "float32"
        self.x_shape = [2, 100]
        self.x_low = -9
        self.x_high = 10
        self.init()

    def test_check_output(self):
        self.check_output(atol=1e-3)


class TestRReluOpTraining(OpTest):
    """
    test the training mode of rrelu op, but 
    set lower to be equal to upper,
    you can subclass this class and modify "setUp" method
    as you want
    """
    def setUp(self):
        self.op_type = "rrelu"
        self.lower = 0.1
        self.fix_seed = True
        self.seed = 1
        self.dtype = "float64"
        self.x_shape = [2, 3, 4, 5]
        self.x_low = -1
        self.x_high = 1
        self.init()

    def init(self):
        x_np = np.random.uniform(self.x_low, self.x_high, self.x_shape).astype(self.dtype)
        out_np = rrelu_inference(x_np, self.lower, self.lower)
        mask_np = np.ones(self.x_shape).astype(self.dtype)
        mask_np[x_np < 0] = self.lower 

        self.inputs = {'X': x_np}
        self.outputs = {'Out': out_np, 'Mask': mask_np}
        self.attrs = {
            'lower': self.lower,
            "upper": self.lower,
            "is_test": False,
            "fix_seed": self.fix_seed,
            "seed": self.seed
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestRReluOpTraining2(TestRReluOpTraining):
    def setUp(self):
        self.op_type = "rrelu"
        self.lower = 0.897
        self.fix_seed = True
        self.seed = 123
        self.dtype = "float64"
        self.x_shape = [11, 4, 5]
        self.x_low = -10
        self.x_high = 10
        self.init()


class TestRReluOpTraining3(TestRReluOpTraining):
    def setUp(self):
        self.op_type = "rrelu"
        self.lower = 0.0786
        self.fix_seed = False
        self.seed = 123
        self.dtype = "float64"
        self.x_shape = [2, 3, 4, 5]
        self.x_low = -100
        self.x_high = 10
        self.init()


class TestRReluOp(OpTest):
    def setUp(self):
        self.op_type = "rrelu"
        self.inputs = {'X': np.random.random((32, 64)).astype("float64")}
        self.attrs = {
            'lower': 0.0, 'upper': 0.8, 
            'fix_seed': False, 'is_test': False}
        self.outputs = {
            'Out': self.inputs['X'],
            'Mask': np.ones((32, 64)).astype("float64")
        }    

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out')


class TestRReluOpInput1d(OpTest):
    def setUp(self):
        self.op_type = "rrelu"
        self.inputs = {'X': np.random.random((2000, )).astype("float64")}
        self.attrs = {
            'lower': 0.2, 'upper': 0.7,
            'fix_seed': True, 'is_test': False}
        self.outputs = {
            'Out': self.inputs['X'],
            'Mask': np.ones((2000)).astype('float64')
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out')


class TestRReluOp2(TestRReluOp):
    def setUp(self):
        self.op_type = "rrelu"
        self.inputs = {'X': np.random.uniform(-100, -10, [19, 3, 4]).astype('float64')}
        self.attrs = {
            'lower': 0, 'upper': 0,
            'fix_seed': True, 'is_test': False}
        self.outputs = {
            'Out': np.zeros([19, 3, 4]).astype('float64'),
            'Mask': np.zeros([19, 3, 4]).astype('float64')
        }


class TestRReluOp3(TestRReluOp):
    def setUp(self):
        self.op_type = "rrelu"
        self.inputs = {'X': np.random.uniform(-10, 10, [2, 30, 4]).astype('float64')}
        self.attrs = {
            'lower': 1, 'upper': 1,
            'fix_seed': False, 'is_test': False}
        self.outputs = {
            'Out': self.inputs['X'],
            'Mask': np.ones([2, 30, 4]).astype('float64')
        }


@skip_check_grad_ci(reason="For inference, check_grad is not required.")
class TestRReluOp9(OpTest):
    def setUp(self):
        self.op_type = "rrelu"
        self.inputs = {'X': np.random.random((32, 64, 3)).astype("float32")}
        self.attrs = {
            'is_test': False
        }
        self.outputs = {'Out': self.inputs['X']}

    def test_check_output(self):
        self.check_output()


@unittest.skipIf(
    not core.is_compiled_with_cuda() or not core.op_support_gpu("rrelu"),
    "core is not compiled with CUDA or core is not support rrelu")
@skip_check_grad_ci(reason="For inference, check_grad is not required.")
class TestFP16RReluOp(OpTest):
    def setUp(self):
        self.op_type = "rrelu"
        self.init_test_case()

        x_np = np.random.uniform(-1, 1, self.x_shape).astype("float16")
        out_np = rrelu_inference(x_np, self.lower, self.upper)
        mask_np = np.ones(self.x_shape).astype("float16")
        mask_np[x_np < 0] = (self.lower + self.upper) / 2.0
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x_np)}
        self.attrs = {
            'lower': self.lower,
            'upper': self.upper,
            'fix_seed': self.fix_seed,
            'is_test': True
        }
        self.outputs = {'Out': out_np, 'Mask': mask_np}

    def init_test_case(self):
        self.x_shape = [32, 64]
        self.lower = 0.17
        self.upper = 0.89
        self.fix_seed = True

    def test_check_output(self):
        self.check_output_with_place(core.CUDAPlace(0), atol=1e-3)


@unittest.skipIf(
    not core.is_compiled_with_cuda() or not core.op_support_gpu("rrelu"),
    "core is not compiled with CUDA or core is not support rrelu")
@skip_check_grad_ci(reason="For inference, check_grad is not required.")
class TestFP16RReluOp2(TestFP16RReluOp):
    def init_test_case(self):
        self.x_shape = [21, 3, 7]
        self.lower = 0.1
        self.upper = 0.127
        self.fix_seed = False


class TestBF16RReluOp(OpTest):
    def setUp(self):
        self.op_type = "rrelu"
        self.dtype = np.uint16
        self.lower = self.upper = 0.78

        x_shape = (32, 64)
        x_np = np.random.uniform(-2, 3, x_shape).astype("float32")
        out_np = rrelu_inference(x_np, self.lower, self.upper)
        mask_np = np.ones(x_shape).astype("float32")
        mask_np[x_np < 0] = self.lower 
        self.inputs = {'X': convert_float_to_uint16(x_np)}
        self.attrs = {
            'lower': self.lower, 'upper': self.upper, 
            'fix_seed': False, 'is_test': False
        }
        self.outputs = {
            'Out': convert_float_to_uint16(out_np),
            'Mask': convert_float_to_uint16(mask_np)
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out')


class TestRReluFAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(fluid.CUDAPlace(0))

    def check_static_result(self, place):
        paddle.enable_static()
        with static.program_guard(static.Program(), static.Program()):
            input = static.data(name="input", shape=[-1, -1], dtype="float32")
            exe = static.Executor(place)
            
            res1 = paddle.nn.functional.rrelu(x=input, lower=1, upper=1, training=False)
            res2 = paddle.nn.functional.rrelu(x=input, lower=1, upper=1, training=True)
            in_np = np.random.uniform(-3, 2, [40, 40]).astype("float32")
            res_np = in_np
            for res in [res1, res2]:
                fetches = exe.run(static.default_main_program(),
                                  feed={"input": in_np},
                                  fetch_list=[res])
                self.assertTrue(np.allclose(fetches[0], res_np))

            lower, upper = 0.17, 0.99
            res3 = paddle.nn.functional.rrelu(x=input, lower=lower, upper=upper, training=False)
            in_np = np.random.uniform(-4, 1, [20, 20]).astype("float32")
            res_np = rrelu_inference(in_np, lower, upper)
            fetches = exe.run(static.default_main_program(),
                              feed={"input": in_np},
                              fetch_list=[res3])
            self.assertTrue(np.allclose(fetches[0], res_np))

            lower = upper = 0.23
            res4 = paddle.nn.functional.rrelu(x=input, lower=lower, upper=upper, training=True)
            in_np = np.random.uniform(-5, 2, [11, 20]).astype("float32")
            res_np = rrelu_inference(in_np, lower, upper)
            fetches = exe.run(static.default_main_program(),
                              feed={"input": in_np},
                              fetch_list=[res4])
            self.assertTrue(np.allclose(fetches[0], res_np))

            # Attention: this part is important!!!
            lower, upper = 0.2, 0.9
            res5 = paddle.nn.functional.rrelu(x=input, lower=lower, upper=upper, training=True)
            in_np = np.random.uniform(-50, 1, [40, 30]).astype("float32")
            fetches = exe.run(static.default_main_program(),
                              feed={"input": in_np},
                              fetch_list=[res5])
            passed_1 = check_element_range_of_rrelu_output_in_training(
                in_np, fetches[0], lower=lower, upper=upper
            )
            self.assertTrue(passed_1)
            passed_2 = check_negative_elements_distribution_of_rrelu_output_in_training(
                in_np, fetches[0], lower=lower, upper=upper, num_segments=5, scale=0.8
            )
            self.assertTrue(passed_2)

        paddle.disable_static()

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        paddle.disable_static()
        for place in self.places:
            in_np = np.random.uniform(-3, 2, [2, 7, 40]).astype("float32")
            res_np = in_np
            in_tensor = paddle.to_tensor(in_np, place=place)
            res1 = paddle.nn.functional.rrelu(x=in_tensor, lower=1, upper=1, training=False)
            res2 = paddle.nn.functional.rrelu(x=in_tensor, lower=1, upper=1, training=True)
            for res in [res1, res2]:
                self.assertTrue(np.allclose(res.numpy(), res_np))

            lower, upper = 0.17, 0.99
            in_np = np.random.uniform(-4, 1, [20, 20]).astype("float32")
            res_np = rrelu_inference(in_np, lower, upper)
            in_tensor = paddle.to_tensor(in_np, place=place)
            res3 = paddle.nn.functional.rrelu(x=in_tensor, lower=lower, upper=upper, training=False)
            self.assertTrue(np.allclose(res3.numpy(), res_np))

            lower = upper = 0.23
            in_np = np.random.uniform(-5, 2, [11, 20]).astype("float32")
            res_np = rrelu_inference(in_np, lower, upper)
            in_tensor = paddle.to_tensor(in_np, place=place)
            res4 = paddle.nn.functional.rrelu(x=in_tensor, lower=lower, upper=upper, training=True)
            self.assertTrue(np.allclose(res4.numpy(), res_np))

            #Attention: this part is important!!!
            lower, upper = 0.23, 0.99
            in_np = np.random.uniform(-50, 1, [11, 20, 3]).astype("float32")
            in_tensor = paddle.to_tensor(in_np, place=place)
            res5 = paddle.nn.functional.rrelu(x=in_tensor, lower=lower, upper=upper, training=True)
            passed_1 = check_element_range_of_rrelu_output_in_training(
                in_np, res5.numpy(), lower=lower, upper=upper
            )
            self.assertTrue(passed_1)
            passed_2 = check_negative_elements_distribution_of_rrelu_output_in_training(
                in_np, res5.numpy(), lower=lower, upper=upper, num_segments=5, scale=0.8
            )
            self.assertTrue(passed_2)


if __name__ == '__main__':
    # paddle.enable_static()
    unittest.main()



class TestRReluFAPIError(unittest.TestCase):
    def test_errors(self):
        paddle.enable_static()
        with static.program_guard(static.Program(), static.Program()):
            def test_Variable():
                # the input of rrelu must be Variable.
                x1 = fluid.create_lod_tensor(
                    np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], fluid.CPUPlace())
                paddle.nn.functional.rrelu(x1, training=True)

            self.assertRaises(TypeError, test_Variable)

            def test_Variable2():
                # the input of rrelu must be Variable.
                x1 = fluid.create_lod_tensor(
                    np.array([-1, -3, 5, 5]), [[1, 1, 1, 1]], fluid.CPUPlace())
                paddle.nn.functional.rrelu(x1, training=False)

            self.assertRaises(TypeError, test_Variable2)

            def test_dtype():
                # the input dtype of dropout must be float32 or float64
                # float16 only can be set on GPU place
                xr = fluid.data(name='xr', shape=[3, 4, 5, 6], dtype="int32")
                paddle.nn.functional.rrelu(xr)

            self.assertRaises(TypeError, test_dtype)

            # I stopped here at 2022/5/4 14:52
            def test_pdtype():
                # p should be int or float
                x2 = fluid.data(name='x2', shape=[3, 4, 5, 6], dtype="float32")
                paddle.nn.functional.dropout(x2, p='0.5')

            self.assertRaises(TypeError, test_pdtype)

            def test_pvalue():
                # p should be 0.<=p<=1.
                x2 = fluid.data(name='x2', shape=[3, 4, 5, 6], dtype="float32")
                paddle.nn.functional.dropout(x2, p=1.2)

            self.assertRaises(ValueError, test_pvalue)

            def test_mode():
                # mode should be 'downscale_in_infer' or 'upscale_in_train'
                x2 = fluid.data(name='x2', shape=[3, 4, 5, 6], dtype="float32")
                paddle.nn.functional.dropout(x2, mode='abc')

            self.assertRaises(ValueError, test_mode)

            def test_axis():
                # axis should be int or list
                x2 = fluid.data(name='x2', shape=[3, 4, 5, 6], dtype="float32")
                paddle.nn.functional.dropout(x2, axis=1.2)

            self.assertRaises(TypeError, test_axis)

            def test_axis_max():
                # maximum of axis should less than dimensions of x
                x2 = fluid.data(name='x2', shape=[3, 4, 5, 6], dtype="float32")
                paddle.nn.functional.dropout(x2, axis=[0, 5])

            self.assertRaises(ValueError, test_axis_max)

            def test_axis_min():
                # minimum of axis should greater equal than 0
                x2 = fluid.data(name='x2', shape=[3, 4, 5, 6], dtype="float32")
                paddle.nn.functional.dropout(x2, axis=[0, -1])

            self.assertRaises(ValueError, test_axis_min)

            def test_axis_len():
                # length of axis should not greater than dimensions of x
                x2 = fluid.data(name='x2', shape=[3, 4, 5, 6], dtype="float32")
                paddle.nn.functional.dropout(x2, axis=[0, 1, 2, 3, 4])

            self.assertRaises(ValueError, test_axis_len)
        paddle.disable_static()

class TestDropoutCAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(fluid.CUDAPlace(0))

    def test_dygraph(self):
        for place in self.places:
            with fluid.dygraph.guard(place):
                input_np = np.random.random([40, 40]).astype("float32")
                result_np = input_np
                input = fluid.dygraph.to_variable(input_np)
                m = paddle.nn.Dropout(p=0.)
                m.eval()
                result = m(input)
                self.assertTrue(np.allclose(result.numpy(), result_np))





class TestDropout2DFAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(fluid.CUDAPlace(0))

    def check_static_result(self, place):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input = fluid.data(
                name="input", shape=[2, 3, 4, 5], dtype="float32")
            res1 = paddle.nn.functional.dropout2d(
                x=input, p=0., training=False, data_format='NCHW')
            res2 = paddle.nn.functional.dropout2d(
                x=input, p=0., training=False, data_format='NHWC')

            in_np = np.random.random([2, 3, 4, 5]).astype("float32")
            res_np = in_np

#             exe = fluid.Executor(place)
#             res_list = [res1, res2]
#             for res in res_list:
#                 fetches = exe.run(fluid.default_main_program(),
#                                   feed={"input": in_np},
#                                   fetch_list=[res])
#                 self.assertTrue(np.allclose(fetches[0], res_np))

#     def test_static(self):
#         for place in self.places:
#             self.check_static_result(place=place)

#     def test_dygraph(self):
#         for place in self.places:
#             with fluid.dygraph.guard(place):
#                 in_np = np.random.random([2, 3, 4, 5]).astype("float32")
#                 res_np = in_np
#                 input = fluid.dygraph.to_variable(in_np)

#                 res1 = paddle.nn.functional.dropout2d(
#                     x=input, p=0., training=False, data_format='NCHW')
#                 res2 = paddle.nn.functional.dropout2d(
#                     x=input, p=0., training=False, data_format='NHWC')

#             res_list = [res1, res2]
#             for res in res_list:
#                 self.assertTrue(np.allclose(res.numpy(), res_np))


# class TestDropout2DFAPIError(unittest.TestCase):
#     def test_errors(self):
#         with program_guard(Program(), Program()):

#             def test_xdim():
#                 # dimentions of x should be 4
#                 x = fluid.data(name='x1', shape=[2, 3, 4, 5, 6], dtype="int32")
#                 paddle.nn.functional.dropout2d(x)

#             self.assertRaises(ValueError, test_xdim)

#             def test_dataformat():
#                 # data_format should be 'NCHW' or 'NHWC'
#                 x = fluid.data(name='x2', shape=[2, 3, 4, 5], dtype="int32")
#                 paddle.nn.functional.dropout2d(x, data_format='CNHW')

#             self.assertRaises(ValueError, test_dataformat)


# class TestDropout2DCAPI(unittest.TestCase):
#     def setUp(self):
#         np.random.seed(123)
#         self.places = [fluid.CPUPlace()]
#         if core.is_compiled_with_cuda():
#             self.places.append(fluid.CUDAPlace(0))

#     def test_dygraph(self):
#         for place in self.places:
#             with fluid.dygraph.guard(place):
#                 input_np = np.random.random([2, 3, 4, 5]).astype("float32")
#                 result_np = input_np
#                 input = fluid.dygraph.to_variable(input_np)
#                 m = paddle.nn.Dropout2D(p=0.)
#                 m.eval()
#                 result = m(input)
#                 self.assertTrue(np.allclose(result.numpy(), result_np))


# class TestDropout3DFAPI(unittest.TestCase):
#     def setUp(self):
#         np.random.seed(123)
#         self.places = [fluid.CPUPlace()]
#         if core.is_compiled_with_cuda():
#             self.places.append(fluid.CUDAPlace(0))

#     def check_static_result(self, place):
#         with fluid.program_guard(fluid.Program(), fluid.Program()):
#             input = fluid.data(
#                 name="input", shape=[2, 3, 4, 5, 6], dtype="float32")
#             res1 = paddle.nn.functional.dropout3d(
#                 x=input, p=0., training=False, data_format='NCDHW')
#             res2 = paddle.nn.functional.dropout3d(
#                 x=input, p=0., training=False, data_format='NDHWC')

#             in_np = np.random.random([2, 3, 4, 5, 6]).astype("float32")
#             res_np = in_np

#             exe = fluid.Executor(place)
#             res_list = [res1, res2]
#             for res in res_list:
#                 fetches = exe.run(fluid.default_main_program(),
#                                   feed={"input": in_np},
#                                   fetch_list=[res])
#                 self.assertTrue(np.allclose(fetches[0], res_np))

#     def test_static(self):
#         for place in self.places:
#             self.check_static_result(place=place)

#     def test_dygraph(self):
#         for place in self.places:
#             with fluid.dygraph.guard(place):
#                 in_np = np.random.random([2, 3, 4, 5, 6]).astype("float32")
#                 res_np = in_np
#                 input = fluid.dygraph.to_variable(in_np)

#                 res1 = paddle.nn.functional.dropout3d(
#                     x=input, p=0., training=False, data_format='NCDHW')
#                 res2 = paddle.nn.functional.dropout3d(
#                     x=input, p=0., training=False, data_format='NDHWC')

#             res_list = [res1, res2]
#             for res in res_list:
#                 self.assertTrue(np.allclose(res.numpy(), res_np))


# class TestDropout3DFAPIError(unittest.TestCase):
#     def test_errors(self):
#         with program_guard(Program(), Program()):

#             def test_xdim():
#                 # dimentions of x should be 5
#                 x = fluid.data(name='x1', shape=[2, 3, 4, 5], dtype="int32")
#                 paddle.nn.functional.dropout3d(x)

#             self.assertRaises(ValueError, test_xdim)

#             def test_dataformat():
#                 # data_format should be 'NCDHW' or 'NDHWC'
#                 x = fluid.data(name='x2', shape=[2, 3, 4, 5, 6], dtype="int32")
#                 paddle.nn.functional.dropout3d(x, data_format='CNDHW')

#             self.assertRaises(ValueError, test_dataformat)


# class TestDropout3DCAPI(unittest.TestCase):
#     def setUp(self):
#         np.random.seed(123)
#         self.places = [fluid.CPUPlace()]
#         if core.is_compiled_with_cuda():
#             self.places.append(fluid.CUDAPlace(0))

#     def test_dygraph(self):
#         for place in self.places:
#             with fluid.dygraph.guard(place):
#                 input_np = np.random.random([2, 3, 4, 5, 6]).astype("float32")
#                 result_np = input_np
#                 input = fluid.dygraph.to_variable(input_np)
#                 m = paddle.nn.Dropout3D(p=0.)
#                 m.eval()
#                 result = m(input)
#                 self.assertTrue(np.allclose(result.numpy(), result_np))


# class TestAlphaDropoutFAPI(unittest.TestCase):
#     def setUp(self):
#         np.random.seed(123)
#         self.places = [fluid.CPUPlace()]
#         if core.is_compiled_with_cuda():
#             self.places.append(fluid.CUDAPlace(0))

#     def check_static_result(self, place):
#         with fluid.program_guard(fluid.Program(), fluid.Program()):
#             input = fluid.data(name="input", shape=[40, 40], dtype="float32")
#             res1 = paddle.nn.functional.alpha_dropout(x=input, p=0.)
#             res2 = paddle.nn.functional.alpha_dropout(
#                 x=input, p=0., training=False)
#             res3 = paddle.nn.functional.alpha_dropout(x=input, p=1.)

#             in_np = np.random.random([40, 40]).astype("float32")
#             res_np = in_np
#             res_np3 = np.zeros_like(in_np)

#             exe = fluid.Executor(place)
#             res_list = [res1, res2]
#             for res in res_list:
#                 fetches = exe.run(fluid.default_main_program(),
#                                   feed={"input": in_np},
#                                   fetch_list=[res])
#                 self.assertTrue(np.allclose(fetches[0], res_np))
#             fetches = exe.run(fluid.default_main_program(),
#                               feed={"input": in_np},
#                               fetch_list=[res3])
#             self.assertTrue(np.allclose(fetches[0], res_np3))

#     def test_static(self):
#         for place in self.places:
#             self.check_static_result(place=place)

#     def test_dygraph(self):
#         for place in self.places:
#             with fluid.dygraph.guard(place):
#                 in_np = np.random.random([40, 40]).astype("float32")
#                 res_np = in_np
#                 res_np3 = np.zeros_like(in_np)
#                 input = fluid.dygraph.to_variable(in_np)

#                 res1 = paddle.nn.functional.alpha_dropout(x=input, p=0.)
#                 res2 = paddle.nn.functional.alpha_dropout(
#                     x=input, p=0., training=False)
#                 res3 = paddle.nn.functional.alpha_dropout(x=input, p=1.)

#             res_list = [res1, res2]
#             for res in res_list:
#                 self.assertTrue(np.allclose(res.numpy(), res_np))
#             self.assertTrue(np.allclose(res3.numpy(), res_np3))


# class TestAlphaDropoutFAPIError(unittest.TestCase):
#     def test_errors(self):
#         with program_guard(Program(), Program()):

#             def test_Variable():
#                 # the input of dropout must be Variable.
#                 x1 = fluid.create_lod_tensor(
#                     np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], fluid.CPUPlace())
#                 paddle.nn.functional.alpha_dropout(x1, p=0.5)

#             self.assertRaises(TypeError, test_Variable)

#             def test_dtype():
#                 # the input dtype of dropout must be float32 or float64
#                 xr = fluid.data(name='xr', shape=[3, 4, 5, 6], dtype="int32")
#                 paddle.nn.functional.alpha_dropout(xr)

#             self.assertRaises(TypeError, test_dtype)

#             def test_pdtype():
#                 # p should be int or float
#                 x2 = fluid.data(name='x2', shape=[3, 4, 5, 6], dtype="float32")
#                 paddle.nn.functional.alpha_dropout(x2, p='0.5')

#             self.assertRaises(TypeError, test_pdtype)

#             def test_pvalue():
#                 # p should be 0.<=p<=1.
#                 x2 = fluid.data(name='x2', shape=[3, 4, 5, 6], dtype="float32")
#                 paddle.nn.functional.alpha_dropout(x2, p=1.2)

#             self.assertRaises(ValueError, test_pvalue)


# class TestAlphaDropoutCAPI(unittest.TestCase):
#     def setUp(self):
#         np.random.seed(123)
#         self.places = [fluid.CPUPlace()]
#         if core.is_compiled_with_cuda():
#             self.places.append(fluid.CUDAPlace(0))

#     def test_dygraph(self):
#         for place in self.places:
#             with fluid.dygraph.guard(place):
#                 input_np = np.random.random([40, 40]).astype("float32")
#                 result_np = input_np
#                 input = fluid.dygraph.to_variable(input_np)
#                 m = paddle.nn.AlphaDropout(p=0.)
#                 m.eval()
#                 result = m(input)
#                 self.assertTrue(np.allclose(result.numpy(), result_np))


# class TestDropoutWithDeterminateSeedGenerator(unittest.TestCase):
#     def setUp(self):
#         paddle.framework.random.set_random_seed_generator('seed0', 123)
#         paddle.framework.random.set_random_seed_generator('seed1', 123)
#         rng0 = paddle.framework.random.get_random_seed_generator('seed0')
#         rng1 = paddle.framework.random.get_random_seed_generator('seed1')
#         self.places = [paddle.CPUPlace()]
#         if paddle.is_compiled_with_cuda():
#             self.places.append(paddle.CUDAPlace(0))

#     def check_static_result(self, place):
#         from paddle.distributed.fleet.meta_parallel.parallel_layers.random import dropout
#         with static.program_guard(static.Program(), static.Program()):
#             input = static.data(name="input", shape=[40, 40], dtype="float32")
#             res1 = dropout(
#                 input,
#                 p=0.3,
#                 training=True,
#                 mode='upscale_in_train',
#                 rng_name='seed0')
#             res2 = dropout(
#                 input,
#                 p=0.3,
#                 training=True,
#                 mode='upscale_in_train',
#                 rng_name='seed1')
#             res3 = dropout(input, p=0.3)

#             in_np = np.random.random([40, 40]).astype("float32")

#             exe = static.Executor(place)
#             res_list = [res1, res2]
#             for i in range(2):
#                 out1, out2 = exe.run(static.default_main_program(),
#                                      feed={"input": in_np},
#                                      fetch_list=res_list)
#                 self.assertTrue(np.allclose(out1, out2))

#     def test_static(self):
#         for place in self.places:
#             self.check_static_result(place=place)


# class TestDropoutBackward(unittest.TestCase):
#     def setUp(self):
#         np.random.seed(123)
#         self.places = [fluid.CPUPlace()]
#         if core.is_compiled_with_cuda():
#             self.places.append(fluid.CUDAPlace(0))

#     def cal_grad_upscale_train(self, mask, prob):
#         return mask.astype("float32") / (1 - prob)

#     def cal_grad_downscale_in_infer(self, mask):
#         return mask.astype("float32")

#     def test_backward_downscale_in_infer(self):
#         _enable_legacy_dygraph()
#         for place in self.places:
#             with fluid.dygraph.guard(place):

#                 input = paddle.uniform([40, 40], dtype="float32")
#                 input.stop_gradient = False
#                 out, mask = core.ops.dropout(input, 'dropout_prob', 0.5)
#                 out.backward()

#                 self.assertTrue(
#                     np.array_equal(input.gradient(
#                     ), self.cal_grad_downscale_in_infer(mask.numpy())))

#     def test_backward_downscale_in_infer_eager(self):
#         for place in self.places:
#             with fluid.dygraph.guard(place):
#                 with _test_eager_guard():
#                     input = paddle.uniform([40, 40], dtype="float32")
#                     input.stop_gradient = False
#                     out, mask = _C_ops.final_state_dropout(
#                         input, None, 0.5, False, "downgrade_in_infer", 0, False)
#                     out.backward()
#                     self.assertTrue(
#                         np.array_equal(input.gradient(
#                         ), self.cal_grad_downscale_in_infer(mask.numpy())))

#     def test_backward_upscale_train(self):
#         _enable_legacy_dygraph()
#         for place in self.places:
#             with fluid.dygraph.guard(place):

#                 prob = 0.5
#                 input = paddle.uniform([40, 40], dtype="float32")
#                 input.stop_gradient = False
#                 out, mask = core.ops.dropout(input, 'dropout_prob', prob,
#                                              "dropout_implementation",
#                                              "upscale_in_train")
#                 out.backward()

#                 self.assertTrue(
#                     np.allclose(input.gradient(
#                     ), self.cal_grad_upscale_train(mask.numpy(), prob)))

#     def test_backward_upscale_train_eager(self):
#         for place in self.places:
#             with fluid.dygraph.guard(place):
#                 with _test_eager_guard():
#                     prob = 0.5
#                     input = paddle.uniform([40, 40], dtype="float32")
#                     input.stop_gradient = False
#                     out, mask = _C_ops.final_state_dropout(
#                         input, None, 0.5, False, "upscale_in_train", 0, False)
#                     out.backward()

#                     self.assertTrue(
#                         np.allclose(input.gradient(
#                         ), self.cal_grad_upscale_train(mask.numpy(), prob)))

#     def test_backward_upscale_train_2(self):
#         _enable_legacy_dygraph()
#         for place in self.places:
#             with fluid.dygraph.guard(place):

#                 prob = 0.3
#                 input = paddle.uniform([40, 40], dtype="float32")
#                 input.stop_gradient = False
#                 out, mask = core.ops.dropout(input, 'dropout_prob', prob,
#                                              "dropout_implementation",
#                                              "upscale_in_train")
#                 out.backward()

#                 self.assertTrue(
#                     np.allclose(input.gradient(
#                     ), self.cal_grad_upscale_train(mask.numpy(), prob)))

#     def test_backward_upscale_train_2_eager(self):
#         for place in self.places:
#             with fluid.dygraph.guard(place):
#                 with _test_eager_guard():

#                     prob = 0.3
#                     input = paddle.uniform([40, 40], dtype="float32")
#                     input.stop_gradient = False
#                     out, mask = _C_ops.final_state_dropout(
#                         input, None, 0.3, False, "upscale_in_train", 0, False)

#                     out.backward()

#                     self.assertTrue(
#                         np.allclose(input.gradient(
#                         ), self.cal_grad_upscale_train(mask.numpy(), prob)))


# class TestRandomValue(unittest.TestCase):
#     def test_fixed_random_number(self):
#         # Test GPU Fixed random number, which is generated by 'curandStatePhilox4_32_10_t'
#         if not paddle.is_compiled_with_cuda():
#             return

#         # Different GPU generate different random value. Only test V100 here.
#         if not "V100" in paddle.device.cuda.get_device_name():
#             return

#         print("Test Fixed Random number on V100 GPU------>")
#         paddle.disable_static()
#         paddle.set_device('gpu')
#         paddle.seed(100)

#         x = paddle.rand([32, 1024, 1024], dtype='float32')
#         out = paddle.nn.functional.dropout(x, 0.25).numpy()
#         index0, index1, index2 = np.nonzero(out)
#         self.assertEqual(np.sum(index0), 390094540)
#         self.assertEqual(np.sum(index1), 12871475125)
#         self.assertEqual(np.sum(index2), 12872777397)
#         self.assertEqual(np.sum(out), 16778744.0)
#         expect = [
#             0.6914956, 0.5294584, 0.19032137, 0.6996228, 0.3338527, 0.8442094,
#             0.96965003, 1.1726775, 0., 0.28037727
#         ]
#         self.assertTrue(np.allclose(out[10, 100, 500:510], expect))

#         x = paddle.rand([32, 1024, 1024], dtype='float64')
#         out = paddle.nn.functional.dropout(x).numpy()
#         index0, index1, index2 = np.nonzero(out)
#         self.assertEqual(np.sum(index0), 260065137)
#         self.assertEqual(np.sum(index1), 8582636095)
#         self.assertEqual(np.sum(index2), 8582219962)
#         self.assertEqual(np.sum(out), 16778396.563660286)
#         expect = [
#             1.28587354, 0.15563703, 0., 0.28799703, 0., 0., 0., 0.54964,
#             0.51355682, 0.33818988
#         ]
#         self.assertTrue(np.allclose(out[20, 100, 500:510], expect))

#         x = paddle.ones([32, 1024, 1024], dtype='float16')
#         out = paddle.nn.functional.dropout(x, 0.75).numpy()
#         index0, index1, index2 = np.nonzero(out)
#         self.assertEqual(np.sum(index0), 130086900)
#         self.assertEqual(np.sum(index1), 4291190105)
#         self.assertEqual(np.sum(index2), 4292243807)
#         expect = [0., 0., 0., 0., 0., 0., 0., 0., 4., 4.]
#         self.assertTrue(np.allclose(out[0, 100, 500:510], expect))

#         paddle.enable_static()


# if __name__ == '__main__':
#     paddle.enable_static()
#     unittest.main()
