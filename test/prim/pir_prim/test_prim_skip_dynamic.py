# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest

import numpy as np

import paddle
from paddle.autograd.ir_backward import grad
from paddle.decomposition import decomp
from paddle.framework import core

paddle.enable_static()


class TestPrimMode(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [2, 1024, 1024]
        self.shape_y = [2, 1024, 1024]
        self.x = np.random.random(self.shape_x).astype("float32")
        self.y = np.random.random(self.shape_y).astype("float32")

    def base_net(self, flag=None):
        if flag == "all":
            core._set_prim_all_enabled(True)
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = paddle.static.data('x', [-1, 1024, 1024], dtype='float32')
            y = paddle.static.data('y', self.shape_y, dtype='float32')
            x.stop_gradient = False
            x1 = paddle.nn.functional.relu(x)
            y.stop_gradient = False
            z = paddle.divide(x1, y)
            res = paddle.nn.functional.gelu(z)
            [res2] = decomp.decompose(main_program, [res])
            gradients = grad(res2, (x, y))
            exe = paddle.static.Executor()
            outs = exe.run(
                feed={
                    'x': self.x,
                    'y': self.y,
                },
                fetch_list=[res2, gradients[0], gradients[1]],
            )

        whole_ops = [op.name() for op in main_program.global_block().ops]
        if flag == "all":
            core._set_prim_all_enabled(False)
            assert 'pd_op.gelu' not in whole_ops
        return outs

    def test_prim_all_dynamic(self):
        os.environ["FLAGS_prim_skip_dynamic"] = "1"
        res_ref = self.base_net()
        res = self.base_net("all")
        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(ref, actual, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
