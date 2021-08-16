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
            axis=self.axis, )

        self.attrs = {
            'mu': mu,
            'use_nesterov': self.use_nesterov,
            'regularization_method': regularization_method,
            'regularization_coeff': regularization_coeff,
            'multi_precision': self.multi_precision,
            'axis': self.axis,
        }

        self.inputs = {
            'Param': param,
            'Velocity': velocity,
            'LearningRate': learning_rate,
            'Grad': grad,
            'Index': index,
        }
        self.outputs = {
            'ParamOut': param_out,
            'VelocityOut': velocity_out,
        }

        if self.multi_precision:
            self.inputs['MasterParam'] = param
            self.outputs['MasterParamOut'] = param_out

    def init_dtype(self):
        pass

    def init_axis(self):
        pass

    def init_multi_precision(self):
        pass

    def init_use_nesterov(self):
        pass

    def test_check_output(self):
        self.check_output()


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


class TestSparseMomentumOpMultiPrecision(unittest.TestCase):
    def setUp(self):
        self.dtype = np.float32
        self.index_dtype = np.int32
        self.axis = 0
        self.multi_precision = True
        self.use_nesterov = False
        self.batch_size = 20
        self.num_classes = 20
        self.init_dtype()
        self.init_axis()
        self.init_multi_precision()
        self.init_use_nesterov()

    def check_with_place(self, place):
        scope = core.Scope()

        param = np.full((self.batch_size, self.num_classes),
                        5.0).astype(self.dtype)
        grad = np.full((self.batch_size, self.num_classes),
                       1.0).astype(self.dtype)
        velocity = np.full((self.batch_size, self.num_classes),
                           1.0).astype(self.dtype)

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

        learning_rate = np.array([2.0]).astype(self.dtype)
        mu = 1.0
        regularization_method = "l2_decay"
        regularization_coeff = 1.0

        param_tensor = scope.var('Param').get_tensor()
        param_tensor.set(param.astype("float16"), place)
        master_param_tensor = scope.var('MasterParam').get_tensor()
        master_param_tensor.set(param, place)

        grad_tensor = scope.var('Grad').get_tensor()
        grad_tensor.set(grad.astype("float16"), place)

        param_out_array = np.full((self.batch_size, self.num_classes),
                                  0.0).astype(self.dtype)
        param_out_tensor = scope.var("ParamOut").get_tensor()
        param_out_tensor.set(param_out_array.astype("float16"), place)
        master_param_out_tensor = scope.var("MasterParamOut").get_tensor()
        master_param_out_tensor.set(param_out_array, place)

        velocity_tensor = scope.var('Velocity').get_tensor()
        velocity_tensor.set(velocity, place)
        velocity_out_array = np.full((self.batch_size, self.num_classes),
                                     0.0).astype(self.dtype)
        velocity_out_tensor = scope.var("VelocityOut").get_tensor()
        velocity_out_tensor.set(velocity_out_array, place)

        index_tensor = scope.var('Index').get_tensor()
        index_tensor.set(index, place)

        lr = scope.var('LearningRate').get_tensor()
        lr.set(learning_rate, place)

        op = Operator(
            "sparse_momentum",
            Param='Param',
            Grad='Grad',
            Velocity='Velocity',
            MasterParam='MasterParam',
            ParamOut='ParamOut',
            VelocityOut='VelocityOut',
            MasterParamOut='MasterParamOut',
            LearningRate='LearningRate',
            Index='Index',
            mu=mu,
            use_nesterov=self.use_nesterov,
            regularization_method=regularization_method,
            regularization_coeff=regularization_coeff,
            multi_precision=True,
            axis=self.axis)
        op.run(scope, place)

        param_out, velocity_out = calculate_sparse_momentum_by_numpy(
            param=param.astype("float16"),
            grad=grad.astype("float16"),
            mu=np.array(
                mu, dtype="float16"),
            velocity=velocity.astype("float16"),
            use_nesterov=self.use_nesterov,
            learning_rate=np.array(
                learning_rate, dtype="float16"),
            regularization_method=regularization_method,
            regularization_coeff=np.array(
                regularization_coeff, dtype="float16"),
            index=index,
            axis=self.axis, )

        self.assertTrue((param_out == np.array(param_out_tensor)).all())
        self.assertTrue((velocity_out == np.array(velocity_out_tensor)).all())

    def test_sparse_momentum(self):
        if core.is_compiled_with_cuda():
            self.check_with_place(fluid.CUDAPlace(0))

    def init_dtype(self):
        pass

    def init_axis(self):
        pass

    def init_multi_precision(self):
        pass

    def init_use_nesterov(self):
        pass


class TestSparseMomentumOpMultiPrecision1(TestSparseMomentumOpMultiPrecision):
    def init_use_nesterov(self):
        self.use_nesterov = True


class TestSparseMomentumOpMultiPrecision2(TestSparseMomentumOpMultiPrecision):
    def init_axis(self):
        self.axis = 1
