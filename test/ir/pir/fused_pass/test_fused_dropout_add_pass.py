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
from paddle.autograd.ir_backward import grad
from paddle.base import core


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "core is not compiled with CUDA",
)
class TestFusedDropoutAdd(unittest.TestCase):
    def test_fused_dropout_add(self):
        with paddle.pir_utils.IrGuard():
            main_program = paddle.static.Program()
            with paddle.static.program_guard(main_program):
                x = paddle.static.data(name="x", shape=[3, 2], dtype="float32")
                y = paddle.static.data(name="y", shape=[3, 2], dtype="float32")
                res1 = paddle.nn.functional.dropout(x=x, p=0.5, training=True)
                res2 = paddle.add(res1, y)
                res3 = paddle.sum(res2)

                op_names = [op.name() for op in main_program.global_block().ops]
                self.assertTrue('pd_op.dropout' in op_names)
                self.assertTrue('pd_op.add' in op_names)
                pm = paddle.pir.PassManager()
                pm.add_pass(
                    'fused_dropout_add_pass', {}
                )  # apply pass to eliminate dead code
                pm.run(main_program)
                op_names = [op.name() for op in main_program.global_block().ops]
                self.assertTrue('pd_op.fused_dropout_add' in op_names)
                self.assertTrue('pd_op.dropout' not in op_names)

                x_np = np.ones([3, 2]).astype("float32")
                y_np = x_np

                exe = paddle.base.Executor(paddle.base.CUDAPlace(0))
                fetches = exe.run(
                    main_program,
                    feed={"x": x_np, "y": y_np},
                    fetch_list=[res3],
                )

    def test_fused_dropout_add_grad(self):
        with paddle.pir_utils.IrGuard():
            main_program = paddle.static.Program()
            with paddle.static.program_guard(main_program):
                x = paddle.static.data(name="x", shape=[3, 2], dtype="float32")
                x.stop_gradient = False
                y = paddle.static.data(name="y", shape=[3, 2], dtype="float32")
                y.stop_gradient = False
                dout = paddle.static.data(
                    name="dout", shape=[3, 2], dtype="float32"
                )
                res0 = paddle.assign(x)
                res1 = paddle.nn.functional.dropout(
                    x=res0, p=0.5, training=True
                )
                res2 = paddle.add(res1, y)
                res3 = paddle.sum(res2)

                # res4 = paddle.incubate.nn.functional.fused_dropout_add( x, y, p=0.5, training=True)
                # res5 = paddle.sum(res4)
                dx = grad(res3, x)

                op_names = [op.name() for op in main_program.global_block().ops]
                self.assertTrue(
                    'pd_op.dropout' in op_names and 'pd_op.add' in op_names
                )
                self.assertTrue(
                    'pd_op.add_grad' in op_names
                    and 'pd_op.dropout_grad' in op_names
                )
                pm = paddle.pir.PassManager()
                pm.add_pass(
                    'fused_dropout_add_pass', {}
                )  # apply pass to eliminate dead code
                pm.run(main_program)
                op_names = [op.name() for op in main_program.global_block().ops]
                self.assertTrue(
                    'pd_op.fused_dropout_add' in op_names
                    and 'pd_op.fused_dropout_add_grad' in op_names
                )
                self.assertTrue(
                    'pd_op.dropout' not in op_names
                    and 'pd_op.dropout_grad' not in op_names
                )

                x_np = np.ones([3, 2]).astype("float32")
                y_np = x_np

                exe = paddle.base.Executor(paddle.base.CUDAPlace(0))
                fetches = exe.run(
                    main_program,
                    feed={"x": x_np, "y": y_np, "dout": y_np},
                    fetch_list=[dx],
                )


if __name__ == "__main__":
    unittest.main()
