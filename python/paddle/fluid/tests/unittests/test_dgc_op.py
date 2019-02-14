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
import paddle.fluid as fluid

g_array_size = 10240


class TestDGCOp(unittest.TestCase):
    def setup(self, place, g_array_size=10240):
        size = g_array_size
        np.random.seed(5)  # fix seed

        self.scope = fluid.global_scope()
        self.place = place
        print("place:", place)

        # numpy data
        # input: U, V, Grad, GradLocal
        self.u_name = "U"
        #self.u = np.zeros(shape=[size], dtype='float32')
        self.u = np.random.random(size).astype("float32")

        self.v_name = "V"
        #self.v = np.zeros(shape=[size], dtype='float32')
        self.v = np.random.random(size).astype("float32")

        self.grad_name = "Grad"
        self.grad = np.random.random(size).astype("float32")

        #self.grad_local_name = "GradLocal"
        #self.grad_local = np.zeros(shape=[size], dtype='float32')

        # output: U_out, V_out, EncodeGrad, GradLocal_out
        self.encode_grad_name = "EncodeGrad"
        #self.encode_grad=np.zeros(shape=[size], dtype='float32')

        # scope data 
        self.u_tensor = self.scope.var(self.u_name).get_tensor()
        self.u_tensor.set(self.u, place)

        self.v_tensor = self.scope.var(self.v_name).get_tensor()
        self.v_tensor.set(self.v, place)

        self.grad_tensor = self.scope.var(self.grad_name).get_tensor()
        self.grad_tensor.set(self.grad, place)

        #self.grad_local_tensor = self.scope.var(self.grad_local_name).get_tensor()
        #self.grad_local_tensor.set(self.grad_local, place)

        self.encode_grad_tensor = self.scope.var(
            self.encode_grad_name).get_tensor()
        #self.encode_grad_tensor.set(self.encode_grad, place)

    def check(self, actual_t, expect_t, place, out_name, atol=1e-5):
        self.assertTrue(
            np.allclose(
                actual_t, expect_t, atol=atol),
            "Output (" + out_name + ") has diff at " + str(place) + "\nExpect "
            + str(expect_t) + "\n" + "But Got" + str(actual_t))

    def test_run_and_check(self):
        self.setup(place=core.CUDAPlace(0))
        #print("data size:", len(g_chunk))
        kwargs = {
            'U': self.u_name,
            'V': self.v_name,
            'Grad': self.grad_name,
            #'GradLocal': self.grad_local_name,
            'U_out': self.u_name,
            'V_out': self.v_name,
            'EncodeGrad': self.encode_grad_name,
            'm': 0.9,
            'ratio': 0.001
        }

        dgc_op = Operator('dgc', **kwargs)

        #atol = 1e-6
        dgc_op.run(self.scope, self.place)

        u_out = np.array(self.u_tensor)
        v_out = np.array(self.v_tensor)
        #grad_local_out = np.array(self.grad_local_tensor)
        encode_grad_out = np.array(self.encode_grad_tensor)

        print("u_out:", u_out[0:20])
        print("v_out:", v_out[0:20])
        #print("grad_local:", grad_local_out[0:20])
        print("encode_grad_out:", encode_grad_out)

        k = int(g_array_size * 0.001)
        index = encode_grad_out[0:k].view(dtype=np.int32)
        value = encode_grad_out[k:2 * k]

        acl = 1e-7

        for i in range(0, k):
            print("idx:", i, "pos:", index[i], "value:", value[i])
            self.assertAlmostEqual(u_out[index[i]], 0.0)
            self.assertAlmostEqual(v_out[index[i]], 0.0)

        a_min = np.amin(value)
        # print("a_min:", a_min)
        dangling = [x for x in v_out if x > a_min]
        print("dangling:", dangling)
        self.assertTrue(len(dangling) == 0)


if __name__ == "__main__":
    unittest.main()
