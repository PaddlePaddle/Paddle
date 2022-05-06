#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
    # use copy of input to avoid changing the value of input in the following calculation
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
    # use copy of input to avoid changing the value of input in the following calculation
    input, op_output = input.copy(), op_output.copy()
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
    # use copy of input to avoid changing the value of input in the following calculation
    input, op_output = input.copy(), op_output.copy()
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


# class TestRReluOpInference(OpTest):
#     """
#     test the inference mode of rrelu op,
#     you can subclass this class and modify "setUp" method
#     as you want
#     """
#     def setUp(self):
#         self.op_type = "rrelu"
#         self.lower = 0.1
#         self.upper = 0.3
#         # self.fix_seed = True
#         # self.seed = 1
#         self.dtype = "float64"
#         self.x_shape = [2, 3, 4, 5]
#         self.x_low = -1
#         self.x_high = 1
#         self.init()

#     def init(self):
#         x_np = np.random.uniform(self.x_low, self.x_high, self.x_shape).astype(self.dtype)
#         out_np = rrelu_inference(x_np, self.lower, self.upper)
#         mask_np = np.ones(self.x_shape).astype(self.dtype)
#         mask_np[x_np < 0] = (self.lower + self.upper) / 2.0

#         self.inputs = {'X': x_np}
#         self.outputs = {'Out': out_np, 'Mask': mask_np}
#         self.attrs = {
#             'lower': self.lower,
#             "upper": self.upper,
#             "is_test": True,
#             # "fix_seed": self.fix_seed,
#             # "seed": self.seed
#         }

#     def test_check_output(self):
#         self.check_output()

#     def test_check_grad(self):
#         self.check_grad(['X'], 'Out')


# class TestRReluOpInference2(TestRReluOpInference):
#     def setUp(self):
#         self.op_type = "rrelu"
#         self.lower = 0.3
#         self.upper = 0.99
#         # self.fix_seed = True
#         # self.seed = 198
#         self.dtype = "float64"
#         self.x_shape = [20, 10]
#         self.x_low = -9
#         self.x_high = -1
#         self.init()


# class TestRReluOpInference3(TestRReluOpInference):
#     def setUp(self):
#         self.op_type = "rrelu"
#         self.lower = 0.8
#         self.upper = 0.99
#         self.fix_seed = False
#         self.seed = 198
#         self.dtype = "float32"
#         self.x_shape = [2, 100]
#         self.x_low = -9
#         self.x_high = 10
#         self.init()

#     def test_check_output(self):
#         self.check_output(atol=1e-3)


# class TestRReluOpTraining(OpTest):
#     """
#     test the training mode of rrelu op, but 
#     set lower to be equal to upper,
#     you can subclass this class and modify "setUp" method
#     as you want
#     """
#     def setUp(self):
#         self.op_type = "rrelu"
#         self.lower = 0.1
#         self.fix_seed = True
#         self.seed = 1
#         self.dtype = "float64"
#         self.x_shape = [2, 3, 4, 5]
#         self.x_low = -1
#         self.x_high = 1
#         self.init()

#     def init(self):
#         x_np = np.random.uniform(self.x_low, self.x_high, self.x_shape).astype(self.dtype)
#         out_np = rrelu_inference(x_np, self.lower, self.lower)
#         mask_np = np.ones(self.x_shape).astype(self.dtype)
#         mask_np[x_np < 0] = self.lower 

#         self.inputs = {'X': x_np}
#         self.outputs = {'Out': out_np, 'Mask': mask_np}
#         self.attrs = {
#             'lower': self.lower,
#             "upper": self.lower,
#             "is_test": False,
#             "fix_seed": self.fix_seed,
#             "seed": self.seed
#         }

#     def test_check_output(self):
#         self.check_output()

#     def test_check_grad(self):
#         self.check_grad(['X'], 'Out')


# class TestRReluOpTraining2(TestRReluOpTraining):
#     def setUp(self):
#         self.op_type = "rrelu"
#         self.lower = 0.897
#         self.fix_seed = True
#         self.seed = 123
#         self.dtype = "float64"
#         self.x_shape = [11, 4, 5]
#         self.x_low = -10
#         self.x_high = 10
#         self.init()


# class TestRReluOpTraining3(TestRReluOpTraining):
#     def setUp(self):
#         self.op_type = "rrelu"
#         self.lower = 0.0786
#         self.fix_seed = False
#         self.seed = 123
#         self.dtype = "float64"
#         self.x_shape = [2, 3, 4, 5]
#         self.x_low = -100
#         self.x_high = 10
#         self.init()


# class TestRReluOp(OpTest):
#     def setUp(self):
#         self.op_type = "rrelu"
#         self.inputs = {'X': np.random.random((32, 64)).astype("float64")}
#         self.attrs = {
#             'lower': 0.0, 'upper': 0.8, 
#             'fix_seed': False, 'is_test': False}
#         self.outputs = {
#             'Out': self.inputs['X'],
#             'Mask': np.ones((32, 64)).astype("float64")
#         }    

#     def test_check_output(self):
#         self.check_output()

#     def test_check_grad_normal(self):
#         self.check_grad(['X'], 'Out')


# class TestRReluOpInput1d(OpTest):
#     def setUp(self):
#         self.op_type = "rrelu"
#         self.inputs = {'X': np.random.random((2000, )).astype("float64")}
#         self.attrs = {
#             'lower': 0.2, 'upper': 0.7,
#             'fix_seed': True, 'is_test': False}
#         self.outputs = {
#             'Out': self.inputs['X'],
#             'Mask': np.ones((2000)).astype('float64')
#         }

#     def test_check_output(self):
#         self.check_output()

#     def test_check_grad_normal(self):
#         self.check_grad(['X'], 'Out')


# class TestRReluOp2(TestRReluOp):
#     def setUp(self):
#         self.op_type = "rrelu"
#         self.inputs = {'X': np.random.uniform(-100, -10, [19, 3, 4]).astype('float64')}
#         self.attrs = {
#             'lower': 0, 'upper': 0,
#             'fix_seed': True, 'is_test': False}
#         self.outputs = {
#             'Out': np.zeros([19, 3, 4]).astype('float64'),
#             'Mask': np.zeros([19, 3, 4]).astype('float64')
#         }


# class TestRReluOp3(TestRReluOp):
#     def setUp(self):
#         self.op_type = "rrelu"
#         self.inputs = {'X': np.random.uniform(-10, 10, [2, 30, 4]).astype('float64')}
#         self.attrs = {
#             'lower': 1, 'upper': 1,
#             'fix_seed': False, 'is_test': False}
#         self.outputs = {
#             'Out': self.inputs['X'],
#             'Mask': np.ones([2, 30, 4]).astype('float64')
#         }


# @skip_check_grad_ci(reason="For inference, check_grad is not required.")
# class TestRReluOp9(OpTest):
#     def setUp(self):
#         self.op_type = "rrelu"
#         self.inputs = {'X': np.random.random((32, 64, 3)).astype("float32")}
#         self.attrs = {
#             'is_test': False
#         }
#         self.outputs = {'Out': self.inputs['X']}

#     def test_check_output(self):
#         self.check_output()

##################################################################3333
#The following tests are passed 
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
            'is_test': True
        }
        self.outputs = {'Out': out_np, 'Mask': mask_np}

    def init_test_case(self):
        self.x_shape = [32, 64]
        self.lower = 0.17
        self.upper = 0.89

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


# class TestBF16RReluOp(OpTest):
#     def setUp(self):
#         self.op_type = "rrelu"
#         self.dtype = np.uint16
#         self.lower = self.upper = 0.78

#         x_shape = (32, 64)
#         x_np = np.random.uniform(-2, 3, x_shape).astype("float32")
#         out_np = rrelu_inference(x_np, self.lower, self.upper)
#         mask_np = np.ones(x_shape).astype("float32")
#         mask_np[x_np < 0] = self.lower 
#         self.inputs = {'X': convert_float_to_uint16(x_np)}
#         self.attrs = {
#             'lower': self.lower, 
#             'upper': self.upper, 
#             'is_test': False
#         }
#         self.outputs = {
#             'Out': convert_float_to_uint16(out_np),
#             'Mask': convert_float_to_uint16(mask_np)
#         }

#     def test_check_output(self):
#         self.check_output()

#     def test_check_grad_normal(self):
#         self.check_grad(['X'], 'Out')


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
                xr = fluid.data(name='xr', shape=[3, 4, 5, 6], dtype="int32")
                paddle.nn.functional.rrelu(xr)

            self.assertRaises(TypeError, test_dtype)

            def test_lower_dtype():
                # lower should be int or float
                x2 = fluid.data(name='x2', shape=[3, 4, 5, 6], dtype="float32")
                paddle.nn.functional.rrelu(x2, lower='0.5', upper=0.8)

            self.assertRaises(TypeError, test_lower_dtype)

            def test_lower_value():
                # lower should be in the interval [0.0, 1.0]
                x2 = fluid.data(name='x2', shape=[3, 4, 5, 6], dtype="float32")
                paddle.nn.functional.rrelu(x2, lower=-0.8, upper=0.5)

            self.assertRaises(ValueError, test_lower_value)

        paddle.disable_static()


class TestRReluCAPI(unittest.TestCase):
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
                rrelu_layer = paddle.nn.RReLU(lower=0.12, upper=0.87)
                rrelu_layer.eval()
                result = rrelu_layer(input)
                self.assertTrue(np.allclose(result.numpy(), result_np))


if __name__ == '__main__':
    # paddle.enable_static()
    unittest.main()
