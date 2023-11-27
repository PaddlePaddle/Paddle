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
from op import Operator

from paddle import base
from paddle.base import core


class TestDGCMomentumOp1(unittest.TestCase):
    def get_tensor(self, name, value, place=None):
        tensor = self.scope.var(name).get_tensor()
        tensor.set(value, self.place if place is None else place)
        return name, tensor

    def setup(self, place, step=0.0):
        self.scope = base.global_scope()
        self.place = place
        print("place:", place)

        self.op_type = "dgc_momentum"
        self.dtype = np.float32
        nranks_val = 2

        param = np.random.random((123, 321)).astype(self.dtype)
        grad = np.random.random((123, 321)).astype(self.dtype)
        velocity = np.zeros((123, 321)).astype(self.dtype)
        learning_rate = np.array([0.001]).astype(self.dtype)
        current_step = np.full((1), step).astype("float32")
        nranks = np.full((1), nranks_val).astype("float32")
        mu = 0.0001
        use_nesterov = False
        rampup_begin_step = 10.0

        # get tensor
        self.param_name, self.param_tensor = self.get_tensor('Param', param)
        self.grad_name, self.grad_tensor = self.get_tensor('Grad', grad)
        self.velocity_name, self.velocity_tensor = self.get_tensor(
            'Velocity', velocity
        )
        self.learning_rate_name, self.learning_rate_tensor = self.get_tensor(
            'LearningRate', learning_rate
        )
        self.current_step_name, self.current_step_tensor = self.get_tensor(
            'current_step', current_step, core.CPUPlace()
        )
        self.nranks_name, self.nranks_tensor = self.get_tensor(
            'nranks', nranks, core.CPUPlace()
        )

        self.kwargs = {
            # inputs
            'Param': self.param_name,
            'Grad': self.grad_name,
            'Velocity': self.velocity_name,
            'LearningRate': self.learning_rate_name,
            'current_step': self.current_step_name,
            'nranks': self.nranks_name,
            # attrs
            'mu': mu,
            'use_nesterov': use_nesterov,
            'rampup_begin_step': rampup_begin_step,
            # outputs
            'ParamOut': self.param_name,
            'VelocityOut': self.velocity_name,
            'Grad_out': self.grad_name,
        }

        velocity_out = mu * velocity + grad / nranks
        if use_nesterov:
            param_out = (
                param - grad * learning_rate - velocity_out * mu * learning_rate
            )
        else:
            param_out = param - learning_rate * velocity_out

        sgd_out = param - learning_rate * grad / nranks

        self.outputs = {
            'ParamOut': param_out,
            'VelocityOut': velocity_out,
            'SGDOut': sgd_out,
        }

    def check(self, actual_t, expect_t, place, out_name, atol=1e-5):
        np.testing.assert_allclose(
            actual_t,
            expect_t,
            rtol=1e-05,
            atol=atol,
            err_msg='Output ('
            + out_name
            + ') has diff at '
            + str(place)
            + '\nExpect '
            + str(expect_t)
            + '\n'
            + 'But Got'
            + str(actual_t),
        )

    def check_momentum_step(self, place):
        self.setup(place=place)

        dgc_momentum_op = Operator(self.op_type, **self.kwargs)
        dgc_momentum_op.run(self.scope, self.place)

        self.check(
            np.array(self.param_tensor),
            self.outputs['ParamOut'],
            self.place,
            self.param_name,
        )

        self.check(
            np.array(self.velocity_tensor),
            self.outputs['VelocityOut'],
            self.place,
            self.velocity_name,
        )

    def check_sgd_step(self, place):
        self.setup(place=place, step=15.0)

        dgc_momentum_op = Operator(self.op_type, **self.kwargs)
        dgc_momentum_op.run(self.scope, self.place)

        self.check(
            np.array(self.param_tensor),
            self.outputs['SGDOut'],
            self.place,
            self.param_name,
        )

    def test_cuda_place(self):
        if not core.is_compiled_with_cuda():
            return
        place = core.CUDAPlace(0)
        self.check_momentum_step(place)
        self.check_sgd_step(place)

    def test_cpu_place(self):
        place = core.CPUPlace()
        self.check_momentum_step(place)
        self.check_sgd_step(place)


if __name__ == "__main__":
    unittest.main()
