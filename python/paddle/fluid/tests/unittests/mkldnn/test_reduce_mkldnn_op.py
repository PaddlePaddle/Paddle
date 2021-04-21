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
from paddle.fluid.tests.unittests.op_test import OpTest, skip_check_grad_ci
import paddle.fluid as fluid
import paddle


class TestReduceSumDefaultOneDNNOp(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.use_mkldnn = True
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float32")}
        self.outputs = {'Out': self.inputs['X'].sum(axis=0)}
        self.attrs = {'use_mkldnn': self.use_mkldnn}

    def test_check_output(self):
        self.check_output()


class TestReduceDefaultWithGradOneDNNOp(TestReduceSumDefaultOneDNNOp):
    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestReduceSum4DOneDNNOp(TestReduceDefaultWithGradOneDNNOp):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.use_mkldnn = True
        self.inputs = {'X': np.random.random((5, 10, 5, 5)).astype("float32")}
        self.attrs = {'use_mkldnn': self.use_mkldnn, 'dim': [2]}
        self.outputs = {
            'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']))
        }


class TestReduceSum4DReduceAllDimAttributeBF16OneDNNOp(
        TestReduceDefaultWithGradOneDNNOp):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.use_mkldnn = True
        self.inputs = {'X': np.random.random((5, 10, 5, 3)).astype("float32")}
        self.attrs = {'use_mkldnn': self.use_mkldnn, 'dim': [0, 1, 2, 3]}
        self.outputs = {
            'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']))
        }


class TestReduceSum5DKeepDimsOneDNNOp(TestReduceDefaultWithGradOneDNNOp):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.use_mkldnn = True
        self.inputs = {'X': np.random.random((2, 5, 3, 2, 2)).astype("float32")}
        self.attrs = {'dim': (2, 3, 4), 'keep_dim': True, 'use_mkldnn': True}
        self.outputs = {
            'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']),
                                        keepdims=self.attrs['keep_dim'])
        }


class TestReduceSum5DReduceAllKeepDimsOneDNNOp(
        TestReduceDefaultWithGradOneDNNOp):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.use_mkldnn = True
        self.inputs = {'X': np.random.random((2, 5, 3, 2, 2)).astype("float32")}
        self.attrs = {'reduce_all': True, 'keep_dim': True, 'use_mkldnn': True}
        self.outputs = {
            'Out': self.inputs['X'].sum(keepdims=self.attrs['keep_dim'])
        }


class TestReduceSum4DReduceAllOneDNNOp(TestReduceDefaultWithGradOneDNNOp):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.use_mkldnn = True
        self.inputs = {'X': np.random.random((5, 6, 2, 10)).astype("float32")}
        self.attrs = {'reduce_all': True, 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': self.inputs['X'].sum()}


@skip_check_grad_ci(
    reason="reduce_max is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework.")
class TestReduceMax3DOneDNNOp(TestReduceSumDefaultOneDNNOp):
    """Remove Max with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_max"
        self.use_mkldnn = True
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float32")}
        self.attrs = {'dim': [-1], 'use_mkldnn': self.use_mkldnn}
        self.outputs = {
            'Out': self.inputs['X'].max(axis=tuple(self.attrs['dim']))
        }


@skip_check_grad_ci(
    reason="reduce_max is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework.")
class TestReduceMax4DNegativeAndPositiveDimsOneDNNOp(
        TestReduceSumDefaultOneDNNOp):
    """Remove Max with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_max"
        self.use_mkldnn = True
        self.inputs = {'X': np.random.random((5, 6, 10, 9)).astype("float32")}
        self.attrs = {'dim': [-1, 0, 1], 'use_mkldnn': self.use_mkldnn}
        self.outputs = {
            'Out': self.inputs['X'].max(axis=tuple(self.attrs['dim']))
        }


@skip_check_grad_ci(
    reason="reduce_min is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework.")
class TestReduceMin3DOneDNNOp(TestReduceSumDefaultOneDNNOp):
    """Remove Min with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_min"
        self.use_mkldnn = True
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float32")}
        self.attrs = {'dim': [2], 'use_mkldnn': self.use_mkldnn}
        self.outputs = {
            'Out': self.inputs['X'].min(axis=tuple(self.attrs['dim']))
        }


class TestReduceMean3DOneDNNOp(TestReduceDefaultWithGradOneDNNOp):
    def setUp(self):
        self.op_type = "reduce_mean"
        self.use_mkldnn = True
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float32")}
        self.attrs = {'dim': [0], 'use_mkldnn': self.use_mkldnn}
        self.outputs = {
            'Out': self.inputs['X'].sum(axis=0) / self.inputs['X'].shape[0]
        }


class TestReduceMean4DReduceAllOneDNNOp(TestReduceDefaultWithGradOneDNNOp):
    def setUp(self):
        self.op_type = "reduce_mean"
        self.use_mkldnn = True
        self.inputs = {'X': np.random.random((5, 6, 8, 10)).astype("float32")}
        self.attrs = {'reduce_all': True, 'use_mkldnn': self.use_mkldnn}
        self.outputs = {
            'Out':
            self.inputs['X'].sum() / np.asarray(self.inputs['X'].shape).prod()
        }


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
