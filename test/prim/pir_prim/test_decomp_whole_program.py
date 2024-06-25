# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.autograd.ir_backward import grad
from paddle.decomposition import decomp

paddle.enable_static()


class TestPrimMode(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [32, 32]
        self.shape_y = [32, 32]
        self.x = np.random.random(self.shape_x).astype("float32")
        self.y = np.random.random(self.shape_y).astype("float32")

    def base_net(self, flag=None):
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = paddle.static.data('x', self.shape_x, dtype='float32')
            y = paddle.static.data('y', self.shape_y, dtype='float32')
            x.stop_gradient = False
            y.stop_gradient = False
            x1 = paddle.sin(x)
            y1 = paddle.cos(y)
            y3 = paddle.matmul(x1, y1)
            tmp1 = paddle.concat((x1, y1, y3))
            tmp1 = paddle.slice(tmp1, axes=[1], starts=[0], ends=[2])
            tmp2 = paddle.mean(tmp1)
            sum_out = paddle.sin(tmp2)
            gradients = grad(sum_out, (x, y))
            if flag == "prim":
                with decomp.prim_guard():
                    decomp.decompose_dist_program(main_program)
            exe = paddle.static.Executor()
            [fwd, dx, dy] = exe.run(
                feed={'x': self.x, 'y': self.y}, fetch_list=[sum_out, gradients]
            )

        whole_ops = [op.name() for op in main_program.global_block().ops]
        if flag == "prim":
            assert 'pd_op.concat_grad' not in whole_ops
        else:
            assert 'pd_op.concat_grad' in whole_ops

        return fwd, dx, dy

    def test_prim_all(self):
        paddle.base.core._set_prim_backward_blacklist("sin_grad", "cos_grad")
        res_ref = self.base_net()
        res = self.base_net("prim")
        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(ref, actual, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
