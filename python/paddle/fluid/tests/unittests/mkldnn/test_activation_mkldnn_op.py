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
from paddle.fluid.tests.unittests.op_test import OpTest
from paddle.fluid.tests.unittests.test_activation_op import TestRelu, TestTanh, TestSqrt, TestAbs
import paddle.fluid as fluid


class TestMKLDNNReluDim2(TestRelu):
    def setUp(self):
        super(TestMKLDNNReluDim2, self).setUp()

        self.attrs = {"use_mkldnn": True}


class TestMKLDNNTanhDim2(TestTanh):
    def setUp(self):
        super(TestMKLDNNTanhDim2, self).setUp()

        self.attrs = {"use_mkldnn": True}


class TestMKLDNNSqrtDim2(TestSqrt):
    def setUp(self):
        super(TestMKLDNNSqrtDim2, self).setUp()

        self.attrs = {"use_mkldnn": True}


class TestMKLDNNAbsDim2(TestAbs):
    def setUp(self):
        super(TestMKLDNNAbsDim2, self).setUp()
        self.attrs = {"use_mkldnn": True}


class TestMKLDNNReluDim4(TestRelu):
    def setUp(self):
        super(TestMKLDNNReluDim4, self).setUp()

        x = np.random.uniform(-1, 1, [2, 4, 3, 5]).astype("float32")
        # The same reason with TestAbs
        x[np.abs(x) < 0.005] = 0.02
        out = np.maximum(x, 0)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {"use_mkldnn": True}


class TestMKLDNNTanhDim4(TestTanh):
    def setUp(self):
        super(TestMKLDNNTanhDim4, self).setUp()

        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 4, 3, 5]).astype("float32")
        }
        self.outputs = {'Out': np.tanh(self.inputs['X'])}
        self.attrs = {"use_mkldnn": True}


class TestMKLDNNSqrtDim4(TestSqrt):
    def setUp(self):
        super(TestMKLDNNSqrtDim4, self).setUp()

        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 4, 3, 5]).astype("float32")
        }
        self.outputs = {'Out': np.sqrt(self.inputs['X'])}
        self.attrs = {"use_mkldnn": True}


class TestMKLDNNAbsDim4(TestAbs):
    def setUp(self):
        super(TestMKLDNNAbsDim4, self).setUp()

        x = np.random.uniform(-1, 1, [2, 4, 3, 5]).astype("float32")
        # The same reason with TestAbs
        x[np.abs(x) < 0.005] = 0.02
        self.inputs = {'X': x}
        self.outputs = {'Out': np.abs(self.inputs['X'])}
        self.attrs = {"use_mkldnn": True}


# Check if primitives already exist in backward
class TestMKLDNNReluPrimitivesAlreadyExist(unittest.TestCase):
    def __assert_close(self, tensor, np_array, msg, atol=1e-4):
        self.assertTrue(np.allclose(np.array(tensor), np_array, atol=atol), msg)

    def test_check_forward_backward(self):
        place = core.CPUPlace()

        np.random.seed(123)
        x = np.random.uniform(-1, 1, [2, 2]).astype(np.float32)
        out = np.abs(x)

        out_grad = np.random.random_sample(x.shape).astype(np.float32)
        x_grad = out_grad * np.sign(x)  # Abs grad calculation

        var_dict = {'x': x, 'out': out, 'out@GRAD': out_grad, 'x@GRAD': x_grad}
        var_names = list(var_dict.keys())
        ground_truth = {name: var_dict[name] for name in var_names}

        program = fluid.Program()
        with fluid.program_guard(program):
            block = program.global_block()
            for name in ground_truth:
                block.create_var(
                    name=name, dtype='float32', shape=ground_truth[name].shape)

            relu_op = block.append_op(
                type="abs",
                inputs={"X": block.var('x'), },
                outputs={"Out": block.var('out')},
                attrs={"use_mkldnn": True})

            # Generate backward op_desc
            grad_op_desc_list, op_grad_to_var = core.get_grad_op_desc(
                relu_op.desc, set(), [])
            grad_op_desc = grad_op_desc_list[0]
            new_op_desc = block.desc.append_op()
            new_op_desc.copy_from(grad_op_desc)
            for var_name in grad_op_desc.output_arg_names():
                block.desc.var(var_name.encode("ascii"))
            grad_op_desc.infer_var_type(block.desc)
            grad_op_desc.infer_shape(block.desc)
            for arg in grad_op_desc.output_arg_names():
                grad_var = block.desc.find_var(arg.encode("ascii"))
                grad_var.set_dtype(core.VarDesc.VarType.FP32)

            exe = fluid.Executor(place)

            # Do at least 2 iterations
            for i in range(2):
                out = exe.run(
                    program,
                    feed={name: var_dict[name]
                          for name in ['x', 'out@GRAD']},
                    fetch_list=['x@GRAD'])

            self.__assert_close(x_grad, out[0], "x@GRAD")


if __name__ == '__main__':
    unittest.main()
