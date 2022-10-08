#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.tests.unittests.op_test import OpTestTool, OpTest, skip_check_grad_ci, convert_float_to_uint16
import paddle.fluid.core as core
import paddle.fluid as fluid
import paddle

paddle.enable_static()


@OpTestTool.skip_if_not_cpu_bf16()
class TestReduceSumDefaultBF16OneDNNOp(OpTest):

    def setUp(self):
        self.op_type = "reduce_sum"
        self.use_mkldnn = True
        self.x_fp32 = np.random.random((5, 6, 10)).astype("float32")
        self.x_bf16 = convert_float_to_uint16(self.x_fp32)
        self.inputs = {'X': self.x_bf16}
        self.outputs = {'Out': self.x_fp32.sum(axis=0)}
        self.attrs = {'use_mkldnn': self.use_mkldnn}

    def test_check_output(self):
        self.check_output(check_dygraph=False)

    def calculate_grads(self):
        tmp_tensor = np.zeros(self.x_fp32.shape).astype("float32")

        prod_of_reduced_dims = self.inputs['X'].shape[0]
        axis = 0

        if "dim" in self.attrs:
            prod_of_reduced_dims = 1
            axis = tuple(self.attrs['dim'])
            for i in range(len(axis)):
                ax = axis[i]
                if axis[i] < 0:
                    ax = len(axis) + axis[i]
                prod_of_reduced_dims *= self.inputs['X'].shape[ax]

        if 'reduce_all' in self.attrs:
            if self.attrs['reduce_all'] is True:
                axis = None
                prod_of_reduced_dims = np.asarray(self.inputs['X'].shape).prod()

        keepdim = False
        if 'keep_dim' in self.attrs:
            keepdim = True

        self.grad_Out = self.x_fp32.sum(axis=axis, keepdims=keepdim)
        self.grad_Out = np.atleast_1d(self.grad_Out)
        self.grad_X = tmp_tensor + self.grad_Out  # broadcast grad

        if self.op_type == 'reduce_mean':
            self.grad_X /= prod_of_reduced_dims


class TestReduceDefaultWithGradBF16OneDNNOp(TestReduceSumDefaultBF16OneDNNOp):

    def test_check_grad(self):
        self.calculate_grads()
        self.check_grad_with_place(
            core.CPUPlace(), ["X"],
            "Out",
            check_dygraph=False,
            user_defined_grads=[self.grad_X],
            user_defined_grad_outputs=[convert_float_to_uint16(self.grad_Out)])


class TestReduceSum4DReduceAllDimAttributeBF16OneDNNOp(
        TestReduceDefaultWithGradBF16OneDNNOp):

    def setUp(self):
        self.op_type = "reduce_sum"
        self.use_mkldnn = True
        self.x_fp32 = np.random.normal(size=(2, 3, 5, 6)).astype('float32')
        self.x_bf16 = convert_float_to_uint16(self.x_fp32)
        self.inputs = {'X': self.x_bf16}
        self.attrs = {'use_mkldnn': self.use_mkldnn, 'dim': [0, 1, 2, 3]}
        self.outputs = {'Out': self.x_fp32.sum(axis=tuple(self.attrs['dim']))}


class TestReduceSum4DReduceAllWithoutReduceAllAttributeNegativeDimsBF16OneDNNOp(
        TestReduceDefaultWithGradBF16OneDNNOp):

    def setUp(self):
        self.op_type = "reduce_sum"
        self.use_mkldnn = True
        self.x_fp32 = np.random.normal(size=(4, 7, 6, 6)).astype('float32')
        self.x_bf16 = convert_float_to_uint16(self.x_fp32)
        self.inputs = {'X': self.x_bf16}
        self.attrs = {'use_mkldnn': self.use_mkldnn, 'dim': [-1, -2, -3, -4]}
        self.outputs = {'Out': self.x_fp32.sum(axis=tuple(self.attrs['dim']))}


class TestReduceSum5DReduceAllKeepDimsBF16OneDNNOp(
        TestReduceDefaultWithGradBF16OneDNNOp):

    def setUp(self):
        self.op_type = "reduce_sum"
        self.use_mkldnn = True
        self.x_fp32 = np.random.normal(size=(2, 5, 3, 2, 5)).astype('float32')
        self.x_bf16 = convert_float_to_uint16(self.x_fp32)
        self.inputs = {'X': self.x_bf16}
        self.attrs = {'reduce_all': True, 'keep_dim': True, 'use_mkldnn': True}
        self.outputs = {'Out': self.x_fp32.sum(keepdims=self.attrs['keep_dim'])}


class TestReduceSum4DReduceAllBF16OneDNNOp(TestReduceDefaultWithGradBF16OneDNNOp
                                           ):

    def setUp(self):
        self.op_type = "reduce_sum"
        self.use_mkldnn = True
        self.x_fp32 = np.random.normal(size=(4, 5, 4, 5)).astype('float32')
        self.x_bf16 = convert_float_to_uint16(self.x_fp32)
        self.inputs = {'X': self.x_bf16}
        self.attrs = {'reduce_all': True, 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': self.x_fp32.sum()}


@skip_check_grad_ci(
    reason="reduce_max is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework.")
class TestReduceMax3DBF16OneDNNOp(TestReduceSumDefaultBF16OneDNNOp):
    """Remove Max with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_max"
        self.use_mkldnn = True
        self.x_fp32 = np.random.random((5, 6, 10)).astype("float32")
        self.x_bf16 = convert_float_to_uint16(self.x_fp32)
        self.inputs = {'X': self.x_bf16}
        self.attrs = {'dim': [-1], 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': self.x_fp32.max(axis=tuple(self.attrs['dim']))}


@skip_check_grad_ci(
    reason="reduce_max is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework.")
class TestReduceMax4DNegativeAndPositiveDimsBF16OneDNNOp(
        TestReduceSumDefaultBF16OneDNNOp):
    """Remove Max with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_max"
        self.use_mkldnn = True
        self.x_fp32 = np.random.random((5, 6, 10, 9)).astype("float32")
        self.x_bf16 = convert_float_to_uint16(self.x_fp32)
        self.inputs = {'X': self.x_bf16}
        self.attrs = {'dim': [-1, 0, 1], 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': self.x_fp32.max(axis=tuple(self.attrs['dim']))}


@skip_check_grad_ci(
    reason="reduce_min is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework.")
class TestReduceMin3DBF16OneDNNOp(TestReduceSumDefaultBF16OneDNNOp):
    """Remove Min with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_min"
        self.use_mkldnn = True
        self.x_fp32 = np.random.random((5, 6, 10)).astype("float32")
        self.x_bf16 = convert_float_to_uint16(self.x_fp32)
        self.inputs = {'X': self.x_bf16}
        self.attrs = {'dim': [2], 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': self.x_fp32.min(axis=tuple(self.attrs['dim']))}


class TestReduceMean3DBF16OneDNNOp(TestReduceDefaultWithGradBF16OneDNNOp):

    def setUp(self):
        self.op_type = "reduce_mean"
        self.use_mkldnn = True
        self.x_fp32 = np.random.random((5, 6, 10)).astype("float32")
        self.x_bf16 = convert_float_to_uint16(self.x_fp32)
        self.inputs = {'X': self.x_bf16}
        self.attrs = {'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': self.x_fp32.sum(axis=0) / self.x_fp32.shape[0]}


class TestReduceMean4DBF16OneDNNOp(TestReduceDefaultWithGradBF16OneDNNOp):

    def setUp(self):
        self.op_type = "reduce_mean"
        self.use_mkldnn = True
        self.x_fp32 = np.random.random((5, 6, 3, 5)).astype("float32")
        self.x_bf16 = convert_float_to_uint16(self.x_fp32)
        self.inputs = {'X': self.x_bf16}
        self.attrs = {'use_mkldnn': self.use_mkldnn, 'dim': [0, 1]}
        self.outputs = {
            'Out':
            self.x_fp32.sum(axis=tuple(self.attrs['dim'])) /
            (self.x_fp32.shape[0] * self.x_fp32.shape[1])
        }


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
