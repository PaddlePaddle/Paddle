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
from pass_test import PassTest

import paddle
from paddle.base import core
from paddle.pir.core import create_parameter

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
    not core.is_compiled_with_cuda() or get_cuda_version() < 11020,
    "weight_only_linear requires CUDA >= 11.2",
)
class TestFusedWeightOnlyLinearPass_Fp32(PassTest):
    def is_config_valid(self, w_shape, bias_shape):
        if w_shape[-1] != bias_shape[0]:
            return False

    def get_valid_op_map(self, dtype, w_shape):
        # weight_quantize need weight's dtype to be fp16 or bf16
        if (
            dtype == "float32"
            or w_shape[0] % 64 != 0
            or w_shape[1] % 16 != 0
            or (
                (
                    paddle.device.cuda.get_device_capability()[0] == 8
                    and paddle.device.cuda.get_device_capability()[1] == 6
                )
                is False
                and (
                    paddle.device.cuda.get_device_capability()[0] == 8
                    and paddle.device.cuda.get_device_capability()[1] == 0
                )
                is False
                and (
                    paddle.device.cuda.get_device_capability()[0] == 7
                    and paddle.device.cuda.get_device_capability()[1] == 5
                )
                is False
                and (
                    paddle.device.cuda.get_device_capability()[0] == 7
                    and paddle.device.cuda.get_device_capability()[1] == 0
                )
                is False
            )
        ):
            self.valid_op_map = {
                "pd_op.weight_only_linear": 0,
                "pd_op.weight_quantize": 0,
                "pd_op.matmul": 1,
                "pd_op.add": 1,
            }
        elif dtype == "float16":
            self.valid_op_map = {
                "pd_op.weight_only_linear": 1,
                "pd_op.weight_quantize": 1,
                "pd_op.matmul": 0,
                "pd_op.add": 0,
            }

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def sample_program(self):
        for dtype in ['float16', "float32"]:
            for w_shape in [[64, 64], [64, 15]]:
                for bias_shape in [[64], [15]]:
                    if self.is_config_valid(w_shape, bias_shape) is False:
                        continue
                    with paddle.pir_utils.IrGuard():
                        start_prog = paddle.static.Program()
                        main_prog = paddle.static.Program()
                        with paddle.pir.core.program_guard(
                            main_prog, start_prog
                        ):
                            x = paddle.static.data(
                                name='x', shape=[3, 64, 64], dtype=dtype
                            )

                            initializer = paddle.nn.initializer.Constant(0.0)
                            w = create_parameter(
                                shape=w_shape,
                                dtype=dtype,
                                initializer=initializer,
                            )
                            bias = paddle.static.data(
                                name="bias",
                                shape=bias_shape,
                                dtype=dtype,
                            )
                            res1 = paddle.matmul(x=x, y=w)
                            out = paddle.add(res1, bias)
                            out = paddle.assign(out)
                            self.pass_list = ['fused_weight_only_linear_pass']
                            self.feeds = {
                                "x": np.random.random((3, 64, 64)).astype(
                                    dtype
                                ),
                                "bias": np.random.random(bias_shape).astype(
                                    dtype
                                ),
                            }
                            self.fetch_list = [out]
                            self.get_valid_op_map(dtype, w_shape)
                            yield [main_prog, start_prog], False

    def test_check_output(self):
        self.check_pass_correct()


if __name__ == "__main__":
    unittest.main()
