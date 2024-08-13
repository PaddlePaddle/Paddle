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

import unittest

import numpy as np

import paddle
from paddle import pir
from paddle.decomposition import decompose
from paddle.framework import core

paddle.enable_static()


def batch_norm_net1(x, r_m, r_v, w, b):
    return paddle.nn.functional.batch_norm(x, r_m, r_v, w, b, training=False)


class TestBuildOp(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [1, 64, 512, 1024]
        self.c_shape = [64]
        self.dtype_x = "float32"
        self.init_x_shape = [1, 64, 512, 1024]
        self.x = np.random.random(self.x_shape).astype(self.dtype_x)
        self.r_m = np.random.random(self.x_shape[1]).astype(self.dtype)
        self.r_v = np.random.random(self.x_shape[1]).astype(self.dtype)
        self.w = np.random.random(self.x_shape[1]).astype(self.dtype)
        self.b = np.random.random(self.x_shape[1]).astype(self.dtype)
        self.net = batch_norm_net1
        self.necessary_ops = "pd_op.batch_norm"
        self.enable_cinn = False
        self.tol = 5e-6

    def get_ir_program(self):
        paddle.enable_static()
        with paddle.pir_utils.OldIrGuard():
            x = paddle.randn([4, 4])
            main_program, start_program = (
                paddle.static.Program(),
                paddle.static.Program(),
            )
            with paddle.static.program_guard(main_program, start_program):
                x = paddle.static.data('x', self.x_shape, x.dtype)
                x.stop_gradients = False
                r_m = paddle.static.data('r_m', self.c_shape, x.dtype)
                r_v = paddle.static.data('r_v', self.c_shape, x.dtype)
                w = paddle.static.data('w', self.c_shape, x.dtype)
                b = paddle.static.data('b', self.c_shape, x.dtype)
                y = batch_norm_net1(x, r_m, r_v, w, b)
                res = paddle.tanh(y)
            pir_program = pir.translate_to_pir(main_program.desc)
            return pir_program

    def test_build_op(self):
        pir_program = self.get_ir_program()
        y = pir_program.global_block().ops[-2].results()
        orig_shape = y[0].shape
        with paddle.pir_utils.IrGuard():
            core._set_prim_forward_enabled(True)
            y_new = decompose(pir_program, y)
            core._set_prim_forward_enabled(False)
            new_shape = y_new[0].shape
            assert (
                orig_shape == new_shape
            ), f"Original shape {orig_shape} is not equal to new shape {new_shape}"
            op_name_list = [op.name() for op in pir_program.global_block().ops]
            assert "pd_op.batch_norm_" not in op_name_list


if __name__ == "__main__":
    unittest.main()
