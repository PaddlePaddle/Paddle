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
from paddle.fluid.op import Operator
from op_test import OpTest


class TestMomentumOp1(OpTest):
    def setUp(self):
        self.op_type = "momentum"
        self.dtype = np.float32
        self.init_dtype()

        param = np.random.random((123, 321)).astype(self.dtype)
        grad = np.random.random((123, 321)).astype(self.dtype)
        velocity = np.zeros((123, 321)).astype(self.dtype)
        learning_rate = np.array([0.001]).astype(self.dtype)
        mu = 0.0001
        use_nesterov = False

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Velocity': velocity,
            'LearningRate': learning_rate
        }

        self.attrs = {'mu': mu}

        velocity_out = mu * velocity + grad
        if use_nesterov:
            param_out = param - grad * learning_rate - \
                        velocity_out * mu * learning_rate
        else:
            param_out = param - learning_rate * velocity_out

        self.outputs = {'ParamOut': param_out, 'VelocityOut': velocity_out}

    def init_dtype(self):
        pass

    def test_check_output(self):
        self.check_output()


class TestMomentumOpFp16(TestMomentumOp1):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output(atol=1e-3)


class TestMomentumOp2(OpTest):
    '''Test Momentum with default values for attributes
    '''

    def setUp(self):
        self.op_type = "momentum"

        param = np.random.random((123, 321)).astype("float32")
        grad = np.random.random((123, 321)).astype("float32")
        velocity = np.zeros((123, 321)).astype("float32")
        learning_rate = np.array([0.001]).astype("float32")
        mu = 0.0001
        use_nesterov = True

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Velocity': velocity,
            'LearningRate': learning_rate
        }

        self.attrs = {'mu': mu, 'use_nesterov': use_nesterov}

        velocity_out = mu * velocity + grad
        if use_nesterov:
            param_out = param - grad * learning_rate - \
                        velocity_out * mu * learning_rate
        else:
            param_out = param - learning_rate * velocity_out

        self.outputs = {'ParamOut': param_out, 'VelocityOut': velocity_out}

    def test_check_output(self):
        self.check_output()


class TestLarsMomentumOp(OpTest):
    def setUp(self):
        self.op_type = "lars_momentum"

        param = np.random.random((123, 321)).astype("float32")
        grad = np.random.random((123, 321)).astype("float32")
        velocity = np.zeros((123, 321)).astype("float32")
        learning_rate = np.array([0.001]).astype("float32")
        mu = 0.0001
        lars_coeff = 0.001
        lars_weight_decay = 0.0005

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Velocity': velocity,
            'LearningRate': learning_rate
        }

        self.attrs = {
            'mu': mu,
            'lars_coeff': lars_coeff,
            'lars_weight_decay': lars_weight_decay
        }

        pnorm = np.sqrt(np.square(param).sum())
        gnorm = np.sqrt(np.square(grad).sum())
        local_lr = learning_rate * lars_coeff * pnorm / (
            gnorm + lars_weight_decay * param)
        velocity_out = mu * velocity + local_lr * (grad + lars_weight_decay *
                                                   param)
        param_out = param - velocity_out

        self.outputs = {'ParamOut': param_out, 'VelocityOut': velocity_out}

    def test_check_output(self):
        self.check_output()


class TestSparseMomentumOp(unittest.TestCase):
    def setUp(self):
        self.use_nesterov = False

    def check_with_place(self, place):
        self.init_kernel()
        scope = core.Scope()
        # create and initialize Grad Variable
        height = 10
        rows = [0, 4, 7]
        row_numel = 12
        mu = 1.0
        use_nesterov = self.use_nesterov

        # create and initialize Param Variable
        param = scope.var('Param').get_tensor()
        param_array = np.full((height, row_numel), 5.0).astype("float32")
        param.set(param_array, place)
        param_out = scope.var("ParamOut").get_tensor()
        param_out_array = np.full((height, row_numel), 0.0).astype("float32")
        param_out.set(param_out_array, place)

        grad_selected_rows = scope.var('Grad').get_selected_rows()
        grad_selected_rows.set_height(height)
        grad_selected_rows.set_rows(rows)
        grad_np_array = np.ones((len(rows), row_numel)).astype("float32")
        grad_np_array[0, 0] = 2.0
        grad_np_array[2, 8] = 4.0
        grad_tensor = grad_selected_rows.get_tensor()
        grad_tensor.set(grad_np_array, place)

        velocity = scope.var('Velocity').get_tensor()
        velocity_np_array = np.ones((height, row_numel)).astype("float32")
        velocity.set(velocity_np_array, place)
        velocity_out = scope.var('VelocityOut').get_tensor()
        velocity_out_np_array = np.full((height, row_numel),
                                        0.0).astype("float32")
        velocity_out.set(velocity_out_np_array, place)

        # create and initialize LeraningRate Variable
        lr = scope.var('LearningRate').get_tensor()
        lr_array = np.full((1), 2.0).astype("float32")
        lr.set(lr_array, place)

        # create and run operator
        op = Operator(
            "momentum",
            Param='Param',
            Grad='Grad',
            Velocity='Velocity',
            ParamOut='ParamOut',
            VelocityOut='VelocityOut',
            LearningRate='LearningRate',
            mu=mu,
            use_nesterov=use_nesterov)
        op.run(scope, place)

        # get and compare result
        param_out_np_array = np.array(param_out)
        velocity_out_np_array = np.array(velocity_out)

        # TODO(dzh): add a more suitable general numpy interface
        # for sparse update.
        _grad_np_array = np.full((height, row_numel), 0.0).astype("float32")
        for i in range(len(rows)):
            _grad_np_array[rows[i]] = grad_np_array[i]
        _velocity_out = mu * velocity_np_array + _grad_np_array
        _param = param_array
        if use_nesterov:
            _param_out = _param - (_grad_np_array + _velocity_out * mu
                                   ) * lr_array
        else:
            _param_out = _param - lr_array * _velocity_out
        self.assertTrue((_velocity_out == velocity_out_np_array).all())
        self.assertTrue((_param_out == param_out_np_array).all())

    def init_kernel(self):
        pass

    def test_sparse_momentum(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            self.check_with_place(place)


class TestSparseMomentumOp2(TestSparseMomentumOp):
    def init_kernel(self):
        self.use_nesterov = True


if __name__ == "__main__":
    unittest.main()
