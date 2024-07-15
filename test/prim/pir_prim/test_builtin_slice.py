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


def meshgrid_net(x1, x2, x3, x4):
    return paddle.meshgrid(x1, x2, x3, x4)


class TestBuildOp(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.c_shape = [64]
        self.init_x_shape = [1, 64, 512, 1024]
        self.x1 = np.random.random(self.c_shape).astype(self.dtype)
        self.x2 = np.random.random(self.c_shape).astype(self.dtype)
        self.x3 = np.random.random(self.c_shape).astype(self.dtype)
        self.x4 = np.random.random(self.c_shape).astype(self.dtype)
        self.net = meshgrid_net

    def get_ir_program(self):
        paddle.enable_static()
        with paddle.pir_utils.OldIrGuard():
            main_program, start_program = (
                paddle.static.Program(),
                paddle.static.Program(),
            )
            with paddle.static.program_guard(main_program, start_program):
                x1 = paddle.static.data('x1', self.c_shape, self.dtype)
                x2 = paddle.static.data('x2', self.c_shape, self.dtype)
                x3 = paddle.static.data('x3', self.c_shape, self.dtype)
                x4 = paddle.static.data('x4', self.c_shape, self.dtype)
                y = meshgrid_net(x1, x2, x3, x4)
                res1 = paddle.tanh(y[0])
                res2 = paddle.sin(y[1])
                res3 = paddle.cos(y[2])
            pir_program = pir.translate_to_pir(main_program.desc)
            return pir_program

    def test_build_op(self):
        pir_program = self.get_ir_program()
        y = pir_program.global_block().ops[-1].results()
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
            assert "pd_op.meshgrid" not in op_name_list


if __name__ == "__main__":
    unittest.main()
