# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.op import Operator
from op_test import OpTest
import paddle
import paddle.fluid as fluid


def calculate_sparse_momentum_by_numpy(param,
                                       grad,
                                       mu,
                                       velocity,
                                       use_nesterov,
                                       learning_rate,
                                       index,
                                       axis,
                                       regularization_method=None,
                                       regularization_coeff=1.0):
    sub_grad = grad.copy()
    grad = np.zeros_like(param)
    if axis == 0:
        unique_index = np.unique(index)
        for idx in unique_index:
            grad[idx, :] = np.sum(sub_grad[index == idx, :], axis=0)
    else:
        unique_index = np.unique(index)
        for idx in unique_index:
            grad[:, idx] = np.sum(sub_grad[:, index == idx], axis=1)
    if regularization_method == "l2_decay":
        grad = grad + regularization_coeff * param

        velocity_out = mu * velocity + grad
        if use_nesterov:
            param_out = param - (grad + velocity_out * mu) * learning_rate
        else:
            param_out = param - learning_rate * velocity_out
    else:
        velocity_out = mu * velocity + grad
        if use_nesterov:
            param_out = param - grad * learning_rate - \
                        velocity_out * mu * learning_rate
        else:
            param_out = param - learning_rate * velocity_out

    return param_out, velocity_out


class TestSparseMomentumOp(OpTest):
    def setUp(self):
        self.op_type = "sparse_momentum"
        self.dtype = np.float32
        self.index_dtype = np.int32
        self.axis = 0
        self.multi_precision = False
        self.use_nesterov = False
        self.batch_size = 20
        self.num_classes = 20
        self.init_dtype()
        self.init_axis()
        self.init_multi_precision()
        self.init_use_nesterov()

        if self.multi_precision:
            assert self.dtype == np.float16

        param = np.random.random(
            (self.batch_size, self.num_classes)).astype(self.dtype)
        grad = np.random.random(
            (self.batch_size, self.num_classes)).astype(self.dtype)
        if self.axis == 0:
            index = np.random.randint(
                0,
                self.batch_size,
                size=(self.batch_size // 2, ),
                dtype=self.index_dtype)
            grad = grad[index]
        else:
            index = np.random.randint(
                0,
                self.num_classes,
                size=(self.num_classes // 2, ),
                dtype=self.index_dtype)
            grad = grad[:, index]
        velocity = np.random.random(
            (self.batch_size, self.num_classes)).astype(self.dtype)
        learning_rate = np.array([0.001]).astype(self.dtype)

        mu = 0.9
        regularization_method = "l2_decay"
        regularization_coeff = 1.0

        param_out, velocity_out = calculate_sparse_momentum_by_numpy(
            param=param,
            grad=grad,
            mu=mu,
            velocity=velocity,
            use_nesterov=self.use_nesterov,
            learning_rate=learning_rate,
            regularization_method=regularization_method,
            regularization_coeff=regularization_coeff,
            index=index,
            axis=self.axis)

        self.attrs = {
            'mu': mu,
            'use_nesterov': self.use_nesterov,
            'regularization_method': regularization_method,
            'regularization_coeff': regularization_coeff,
            'multi_precision': self.multi_precision,
            'axis': self.axis,
        }

        self.inputs = {
            'Param': param.astype("float16") if self.multi_precision else param,
            'Velocity': velocity.astype("float32")
            if self.multi_precision else velocity,
            'LearningRate': learning_rate.astype("float32")
            if self.multi_precision else learning_rate,
            'Grad': grad.astype("float16") if self.multi_precision else grad,
            'Index': index,
            'Axis': np.array(self.axis).astype(np.int32),
        }
        self.outputs = {
            'ParamOut': param_out.astype("float16")
            if self.multi_precision else param_out,
            'VelocityOut': velocity_out.astype("float32")
            if self.multi_precision else velocity_out,
        }

        if self.multi_precision:
            self.inputs['MasterParam'] = param.astype(
                "float32") if self.multi_precision else param
            self.outputs['MasterParamOut'] = param_out.astype(
                "float32") if self.multi_precision else param_out

    def init_dtype(self):
        pass

    def init_axis(self):
        pass

    def init_multi_precision(self):
        pass

    def init_use_nesterov(self):
        pass

    def test_check_output(self):
        self.check_output(
            atol=5e-3 if self.multi_precision else 1e-5, check_eager=True)


class TestSparseMomentumOpDtype1(TestSparseMomentumOp):
    def init_dtype(self):
        self.dtype = np.float32
        self.index_dtype = np.int64


class TestSparseMomentumOpDtype2(TestSparseMomentumOp):
    def init_dtype(self):
        self.dtype = np.float64
        self.index_dtype = np.int32


class TestSparseMomentumOpDtype3(TestSparseMomentumOp):
    def init_dtype(self):
        self.dtype = np.float64
        self.index_dtype = np.int64


class TestSparseMomentumOpAxis(TestSparseMomentumOp):
    def init_axis(self):
        self.axis = 1


class TestSparseMomentumOpNesterov(TestSparseMomentumOp):
    def init_use_nesterov(self):
        self.use_nesterov = True


class TestSparseMomentumOpMultiPrecision(TestSparseMomentumOp):
    def init_dtype(self):
        self.dtype = np.float16
        self.index_dtype = np.int32

    def init_multi_precision(self):
        self.multi_precision = True

    def init_use_nesterov(self):
        self.use_nesterov = True


class TestSparseMomentumOpMultiPrecision1(TestSparseMomentumOp):
    def init_dtype(self):
        self.dtype = np.float16
        self.index_dtype = np.int64

    def init_multi_precision(self):
        self.multi_precision = True

    def init_use_nesterov(self):
        self.use_nesterov = True


class TestSparseMomentumOpMultiPrecision2(TestSparseMomentumOp):
    def init_dtype(self):
        self.dtype = np.float16
        self.index_dtype = np.int32

    def init_multi_precision(self):
        self.multi_precision = True

    def init_use_nesterov(self):
        self.use_nesterov = False


class TestSparseMomentumOpMultiPrecision3(TestSparseMomentumOp):
    def init_dtype(self):
        self.dtype = np.float16
        self.index_dtype = np.int64

    def init_multi_precision(self):
        self.multi_precision = True

    def init_use_nesterov(self):
        self.use_nesterov = False


if __name__ == "__main__":
    unittest.main()
