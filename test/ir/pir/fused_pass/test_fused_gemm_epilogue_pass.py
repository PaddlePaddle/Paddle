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
from paddle.autograd.ir_backward import grad as ir_grad
from paddle.base import core

np.random.seed(2013)

import os
import re


def get_cuda_version():
    result = os.popen("nvcc --version").read()
    regex = r'release (\S+),'
    match = re.search(regex, result)
    if match:
        num = str(match.group(1))
        integer, decimal = num.split('.')
        return int(integer) * 1000 + int(float(decimal) * 10)
    else:
        return -1


@unittest.skipIf(
    not core.is_compiled_with_cuda() or get_cuda_version() < 11060,
    "core is not complied with CUDA or nvcc version is less than11.6",
)
class TestFusedgemm_epilogueAdd(unittest.TestCase):
    def test_fused_gemm_epilogue_add(self):
        with paddle.pir_utils.IrGuard():
            x_np = np.random.normal(3, 2.5, size=(1024, 1024)).astype(
                np.float32
            )
            y_np = x_np
            z_np = np.random.normal(3, 2.5, size=(1024)).astype(np.float32)
            main_program = paddle.static.Program()
            with paddle.static.program_guard(main_program):
                x_ = paddle.static.data(
                    name="x", shape=[1024, 1024], dtype="float32"
                )
                y_ = paddle.static.data(
                    name="y", shape=[1024, 1024], dtype="float32"
                )
                z_ = paddle.static.data(name="z", shape=[1024], dtype="float32")
                x_.stop_gradient = False
                y_.stop_gradient = False
                z_.stop_gradient = False
                x = paddle.assign(x_)
                y = paddle.assign(y_)
                z = paddle.assign(z_)
                res1 = paddle.matmul(x=x, y=y)
                res2 = paddle.add(res1, z)
                res3 = paddle.assign(res2)

                res4, res5, res6 = ir_grad(res3, [x, y, z])
                res4_ = paddle.assign(res4)
                res5_ = paddle.assign(res5)
                res6_ = paddle.assign(res6)
                op_names = [op.name() for op in main_program.global_block().ops]
                self.assertTrue(
                    'pd_op.matmul' in op_names and 'pd_op.add' in op_names
                )
                self.assertTrue(
                    'pd_op.add_grad' in op_names
                    and 'pd_op.matmul_grad' in op_names
                )

                with paddle.static.scope_guard(paddle.static.Scope()):
                    exe = paddle.base.Executor(paddle.base.CUDAPlace(0))
                    fetches0 = exe.run(
                        main_program,
                        feed={"x": x_np, "y": y_np, "z": z_np},
                        fetch_list=[res3, res4_, res5_, res6_],
                    )
                # main_program = main_program.clone()

                pm = paddle.pir.PassManager()
                pm.add_pass(
                    'fused_gemm_epilogue_pass'
                )  # apply pass to elimitate dead code
                pm.run(main_program)
                op_names = [op.name() for op in main_program.global_block().ops]
                self.assertTrue(
                    'pd_op.fused_gemm_epilogue' in op_names
                    and 'pd_op.fused_gemm_epilogue_grad' in op_names
                )

                with paddle.static.scope_guard(paddle.static.Scope()):
                    exe = paddle.base.Executor(paddle.base.CUDAPlace(0))
                    fetches1 = exe.run(
                        main_program,
                        feed={"x": x_np, "y": y_np, "z": z_np},
                        fetch_list=[res3, res4_, res5_, res6_],
                    )

                np.array_equal(fetches0[0], fetches1[0])
                np.array_equal(fetches0[1], fetches1[1])
                np.array_equal(fetches0[2], fetches1[2])
                np.array_equal(fetches0[3], fetches1[3])


if __name__ == "__main__":
    unittest.main()
