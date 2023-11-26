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
    core.is_compiled_with_cuda() and get_cuda_version() < 11020,
    "weight_only_linear needs CUDA version greater or euqal than 11.2",
)
class TestMatmulToWeightOnly(unittest.TestCase):
    def test_matmul_to_weight_only(self):
        with paddle.pir_utils.IrGuard():
            x_np = np.random.normal(3, 2.5, size=(3, 64, 64)).astype(np.float32)
            w_np = np.random.normal(3, 2.5, size=(64, 64)).astype(np.float32)
            bias_np = np.random.normal(3, 2.5, size=(64)).astype(np.float32)
            main_program = paddle.static.Program()
            with paddle.static.program_guard(main_program):
                x = paddle.static.data(
                    name="x", shape=[3, 64, 64], dtype="float32"
                )
                w = paddle.static.data(
                    name="w", shape=[64, 64], dtype="float32"
                )
                bias_ = paddle.static.data(
                    name="bias", shape=[64], dtype="float32"
                )
                bias = paddle.assign(bias_)
                res1 = paddle.matmul(x=x, y=w)
                res2 = paddle.add(res1, bias)

                op_names = [op.name() for op in main_program.global_block().ops]
                self.assertTrue('pd_op.matmul' in op_names)
                self.assertTrue('pd_op.add' in op_names)

                with paddle.static.scope_guard(paddle.static.Scope()):
                    exe = paddle.base.Executor(paddle.base.CUDAPlace(0))
                    fetches = exe.run(
                        main_program,
                        feed={"x": x_np, "w": w_np, "bias": bias_np},
                        fetch_list=[res2],
                    )
                pm = paddle.pir.PassManager()
                pm.add_pass('matmul_to_weight_only_linear_pass')
                pm.run(main_program)
                op_names = [op.name() for op in main_program.global_block().ops]
                self.assertTrue('pd_op.weight_only_linear' in op_names)


if __name__ == "__main__":
    unittest.main()
