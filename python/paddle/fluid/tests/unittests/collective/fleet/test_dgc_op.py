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
import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle.fluid as fluid

g_array_size = 102400


class TestDGCOp(unittest.TestCase):

    def setup(self, place, array_size=g_array_size):
        size = array_size
        np.random.seed(5)  # fix seed

        self.scope = fluid.global_scope()
        self.place = place
        print("place:", place)

        # numpy data
        # inputs: U, V, Grad, current_step, nranks
        self.u_name = "U"
        self.u = np.random.random(size).astype("float32")

        self.v_name = "V"
        self.v = np.random.random(size).astype("float32")

        self.grad_name = "Grad"
        self.grad = np.random.random(size).astype("float32")

        self.param_name = "Param"
        self.param = np.random.random(size).astype("float32")

        self.current_step_name = "current_step"
        self.current_step = np.full((1), 0.0).astype("float32")

        self.nranks_name = "nranks"
        self.nranks = np.full((1), 2.0).astype("float32")

        # output: U_out, V_out, EncodeGrad, GradLocal_out, k, GatherBuff
        self.encode_grad_name = "EncodeGrad"
        self.k_name = "k"
        self.k = np.full((1), 0.0).astype("float32")
        self.gather_buff_name = "GatherBuff"

        # scope data
        self.u_tensor = self.scope.var(self.u_name).get_tensor()
        self.u_tensor.set(self.u, place)

        self.v_tensor = self.scope.var(self.v_name).get_tensor()
        self.v_tensor.set(self.v, place)

        self.grad_tensor = self.scope.var(self.grad_name).get_tensor()
        self.grad_tensor.set(self.grad, place)

        self.param_tensor = self.scope.var(self.param_name).get_tensor()
        self.param_tensor.set(self.param, place)

        self.current_step_tensor = self.scope.var(
            self.current_step_name).get_tensor()
        self.current_step_tensor.set(self.current_step, core.CPUPlace())

        self.nranks_tensor = self.scope.var(self.nranks_name).get_tensor()
        self.nranks_tensor.set(self.nranks, core.CPUPlace())

        self.encode_grad_tensor = self.scope.var(
            self.encode_grad_name).get_tensor()

        self.k_tensor = self.scope.var(self.k_name).get_tensor()
        self.k_tensor.set(self.k, core.CPUPlace())

        self.gather_buff_tensor = self.scope.var(
            self.gather_buff_name).get_tensor()

    def check(self, actual_t, expect_t, place, out_name, atol=1e-5):
        np.testing.assert_allclose(
            actual_t,
            expect_t,
            rtol=1e-05,
            atol=atol,
            err_msg='Output (' + out_name + ') has diff at ' + str(place) +
            '\nExpect ' + str(expect_t) + '\n' + 'But Got' + str(actual_t))

    def test_run_and_check(self):
        self.setup(place=core.CUDAPlace(0))
        kwargs = {
            # inputs
            'U': self.u_name,
            'V': self.v_name,
            'Grad': self.grad_name,
            'Param': self.param_name,
            'current_step': self.current_step_name,
            'nranks': self.nranks_name,

            # outputs
            'U_out': self.u_name,
            'V_out': self.v_name,
            'EncodeGrad': self.encode_grad_name,
            'Grad_out': self.grad_name,
            'k': self.k_name,
            'GatherBuff': self.gather_buff_name,

            # attrs
            'm': 0.9,
            'sparsity': [0.75, 0.9375, 0.984375, 0.996, 0.999],
            'use_nesterov': True,
            'rampup_begin_step': float(0.0),
            'rampup_step': float(10.0),
            'regular_coeff': float(1e-4),
            'regular_type': int(2),
        }

        dgc_op = Operator('dgc', **kwargs)

        #atol = 1e-6
        dgc_op.run(self.scope, self.place)

        u_out = np.array(self.u_tensor)
        v_out = np.array(self.v_tensor)
        grad_out = np.array(self.grad_tensor)
        encode_grad_out = np.array(self.encode_grad_tensor)
        k = int(np.array(self.k_tensor)[0])

        print("u_out:", u_out[0:20])
        print("v_out:", v_out[0:20])
        print("encode_grad_out:", encode_grad_out)
        print("k_out:", k)

        self.assertEqual(k, int(g_array_size * 0.25))

        index = encode_grad_out[0:k].view(dtype=np.int32)
        value = encode_grad_out[k:2 * k]

        acl = 1e-7

        for i in range(0, k):
            self.assertAlmostEqual(u_out[index[i]], 0.0)
            self.assertAlmostEqual(v_out[index[i]], 0.0)

        a_min = np.amin(value)
        dangling = [x for x in v_out if x > a_min]


if __name__ == "__main__":
    unittest.main()
