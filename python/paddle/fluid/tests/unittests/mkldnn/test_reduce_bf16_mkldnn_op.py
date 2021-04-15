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

from __future__ import print_function

import unittest
import numpy as np
from paddle.fluid.tests.unittests.op_test import OpTest, skip_check_grad_ci, convert_float_to_uint16
import paddle.fluid.core as core
import paddle.fluid as fluid
import paddle


@unittest.skipIf(not core.supports_bfloat16(),
                 "place does not support BF16 evaluation")
@unittest.skipIf(core.is_compiled_with_cuda(),
                 "core is compiled with CUDA which has no BF implementation")
@skip_check_grad_ci(reason="not implemented")
class TestReduceSumDefaultBF16ONEDNNOp(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.use_mkldnn = True
        x_fp32 = np.random.random((5, 6, 10)).astype("float32")
        x_bf16 = convert_float_to_uint16(x_fp32)
        self.inputs = {'X': x_bf16}
        self.outputs = {'Out': x_fp32.sum(axis=0)}
        self.attrs = {'use_mkldnn': self.use_mkldnn}

    def test_check_output(self):
        self.check_output(check_dygraph=False)


@skip_check_grad_ci(reason="not implemented")
class TestReduceSum4DBF16ONEDNNOp(TestReduceSumDefaultBF16ONEDNNOp):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.use_mkldnn = True
        x_fp32 = np.random.random((5, 10, 5, 5)).astype("float32")
        x_bf16 = convert_float_to_uint16(x_fp32)
        self.inputs = {'X': x_bf16}
        self.attrs = {'use_mkldnn': self.use_mkldnn, 'dim': [2]}
        self.outputs = {'Out': x_fp32.sum(axis=tuple(self.attrs['dim']))}


@skip_check_grad_ci(reason="not implemented")
class TestReduceSum4DReduceAllWithoutReduceAllAttributeBF16ONEDNNOp(
        TestReduceSumDefaultBF16ONEDNNOp):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.use_mkldnn = True
        x_fp32 = np.random.normal(size=(2, 3, 5, 6)).astype('float32')
        x_bf16 = convert_float_to_uint16(x_fp32)
        self.inputs = {'X': x_bf16}
        self.attrs = {'use_mkldnn': self.use_mkldnn, 'dim': [0, 1, 2, 3]}
        self.outputs = {'Out': x_fp32.sum(axis=tuple(self.attrs['dim']))}


@skip_check_grad_ci(reason="not implemented")
class TestReduceSum4DReduceAllWithoutReduceAllAttributeNegativeDimsBF16ONEDNNOp(
        TestReduceSumDefaultBF16ONEDNNOp):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.use_mkldnn = True
        x_fp32 = np.random.normal(size=(2, 7, 3, 5)).astype('float32')
        x_bf16 = convert_float_to_uint16(x_fp32)
        self.inputs = {'X': x_bf16}
        self.attrs = {'use_mkldnn': self.use_mkldnn, 'dim': [-1, -2, -3, -4]}
        self.outputs = {'Out': x_fp32.sum(axis=tuple(self.attrs['dim']))}


@skip_check_grad_ci(reason="not implemented")
class TestReduceSum5DKeepDimsONEDNNOp(TestReduceSumDefaultBF16ONEDNNOp):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.use_mkldnn = True
        x_fp32 = np.random.random((2, 5, 3, 2, 2)).astype("float32")
        x_bf16 = convert_float_to_uint16(x_fp32)
        self.inputs = {'X': x_bf16}
        self.attrs = {'dim': (2, 3, 4), 'keep_dim': True, 'use_mkldnn': True}
        self.outputs = {
            'Out': x_fp32.sum(axis=tuple(self.attrs['dim']),
                              keepdims=self.attrs['keep_dim'])
        }


@skip_check_grad_ci(reason="not implemented")
class TestReduceSum5DReduceAllKeepDimsBF16ONEDNNOp(
        TestReduceSumDefaultBF16ONEDNNOp):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.use_mkldnn = True
        x_fp32 = np.random.normal(size=(2, 5, 3, 2, 4)).astype('float32')
        x_bf16 = convert_float_to_uint16(x_fp32)
        self.inputs = {'X': x_bf16}
        self.attrs = {'reduce_all': True, 'keep_dim': True, 'use_mkldnn': True}
        self.outputs = {'Out': x_fp32.sum(keepdims=self.attrs['keep_dim'])}


@skip_check_grad_ci(reason="not implemented")
class TestReduceSum4DReduceAllBF16ONEDNNOp(TestReduceSumDefaultBF16ONEDNNOp):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.use_mkldnn = True
        x_fp32 = np.random.normal(size=(4, 3, 2, 3)).astype('float32')
        x_bf16 = convert_float_to_uint16(x_fp32)
        self.inputs = {'X': x_bf16}
        self.attrs = {'reduce_all': True, 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': x_fp32.sum()}


@skip_check_grad_ci(
    reason="reduce_max is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework.")
class TestReduceMax3DBF16ONEDNNOp(TestReduceSumDefaultBF16ONEDNNOp):
    """Remove Max with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_max"
        self.use_mkldnn = True
        x_fp32 = np.random.random((5, 6, 10)).astype("float32")
        x_bf16 = convert_float_to_uint16(x_fp32)
        self.inputs = {'X': x_bf16}
        self.attrs = {'dim': [-1], 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': x_fp32.max(axis=tuple(self.attrs['dim']))}


@skip_check_grad_ci(
    reason="reduce_max is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework.")
class TestReduceMax4DNegativeAndPositiveDimsBF16ONEDNNOp(
        TestReduceSumDefaultBF16ONEDNNOp):
    """Remove Max with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_max"
        self.use_mkldnn = True
        x_fp32 = np.random.random((5, 6, 10, 9)).astype("float32")
        x_bf16 = convert_float_to_uint16(x_fp32)
        self.inputs = {'X': x_bf16}
        self.attrs = {'dim': [-1, 0, 1], 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': x_fp32.max(axis=tuple(self.attrs['dim']))}


@skip_check_grad_ci(
    reason="reduce_min is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework.")
class TestReduceMin3DBF16ONEDNNOp(TestReduceSumDefaultBF16ONEDNNOp):
    """Remove Min with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_min"
        self.use_mkldnn = True
        x_fp32 = np.random.random((5, 6, 10)).astype("float32")
        x_bf16 = convert_float_to_uint16(x_fp32)
        self.inputs = {'X': x_bf16}
        self.attrs = {'dim': [2], 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': x_fp32.min(axis=tuple(self.attrs['dim']))}


@skip_check_grad_ci(reason="not implemented")
class TestReduceMean3DBF16ONEDNNOp(TestReduceSumDefaultBF16ONEDNNOp):
    def setUp(self):
        self.op_type = "reduce_mean"
        self.use_mkldnn = True
        x_fp32 = np.random.random((5, 6, 10)).astype("float32")
        x_bf16 = convert_float_to_uint16(x_fp32)
        self.inputs = {'X': x_bf16}
        self.attrs = {'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': x_fp32.sum(axis=0) / x_fp32.shape[0]}


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
