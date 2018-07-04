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
from op_test import OpTest


class TestSumOp(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].sum(axis=0)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestMeanOp(OpTest):
    def setUp(self):
        self.op_type = "reduce_mean"
        self.inputs = {'X': np.random.random((5, 6, 2, 10)).astype("float64")}
        self.attrs = {'dim': [1]}
        self.outputs = {
            'Out': self.inputs['X'].mean(axis=tuple(self.attrs['dim']))
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestMaxOp(OpTest):
    """Remove Max with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_max"
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.attrs = {'dim': [-1]}
        self.outputs = {
            'Out': self.inputs['X'].max(axis=tuple(self.attrs['dim']))
        }

    def test_check_output(self):
        self.check_output()


class TestMinOp(OpTest):
    """Remove Min with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_min"
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.attrs = {'dim': [2]}
        self.outputs = {
            'Out': self.inputs['X'].min(axis=tuple(self.attrs['dim']))
        }

    def test_check_output(self):
        self.check_output()


class TestProdOp(OpTest):
    def setUp(self):
        self.op_type = "reduce_prod"
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].prod(axis=0)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestKeepDimReduce(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.attrs = {'dim': [-2], 'keep_dim': True}
        self.outputs = {
            'Out':
            self.inputs['X'].sum(axis=tuple(self.attrs['dim']), keepdims=True)
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class Test1DReduce(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.inputs = {'X': np.random.random(20).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].sum(axis=0)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestReduceAll(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.inputs = {'X': np.random.random((5, 6, 2, 10)).astype("float64")}
        self.attrs = {'reduce_all': True}
        self.outputs = {'Out': self.inputs['X'].sum()}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


## reduction in multi dims
class TestReduceMeanOpMultiAxises(OpTest):
    def setUp(self):
        self.op_type = "reduce_mean"
        self.inputs = {'X': np.random.random((5, 6, 2, 10)).astype("float64")}
        self.attrs = {'dim': [1, 2]}
        self.outputs = {'Out': self.inputs['X'].mean(axis=(1, 2))}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestReduceMaxOpMultiAxises(OpTest):
    """Remove Max with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_max"
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.attrs = {'dim': [-2, -1]}
        self.outputs = {
            'Out': self.inputs['X'].max(axis=tuple(self.attrs['dim']))
        }

    def test_check_output(self):
        self.check_output()


class TestReduceMinOpMultiAxises(OpTest):
    """Remove Min with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_min"
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.attrs = {'dim': [1, 2]}
        self.outputs = {
            'Out': self.inputs['X'].min(axis=tuple(self.attrs['dim']))
        }

    def test_check_output(self):
        self.check_output()


class TestKeepDimReduceSumMultiAxises(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.attrs = {'dim': [-2, -1], 'keep_dim': True}
        self.outputs = {
            'Out':
            self.inputs['X'].sum(axis=tuple(self.attrs['dim']), keepdims=True)
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


if __name__ == '__main__':
    unittest.main()
