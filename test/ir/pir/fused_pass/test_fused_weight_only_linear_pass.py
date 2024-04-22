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
import re
import unittest

import numpy as np
from pass_test import PassTest

import paddle
from paddle.base import core
from paddle.pir.core import create_parameter

np.random.seed(2013)


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
class TestFusedWeightOnlyLinearPass_WithBias(PassTest):
    def is_config_valid(self, w_shape, bias_shape):
        if w_shape[-1] != bias_shape[-1]:
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
        self.pass_attr_list = [{'fused_weight_only_linear_pass': {}}]

    def sample_program(self):
        for dtype in ['float16', "float32"]:
            for w_shape in [[4096, 2048], [4096, 1024]]:
                for bias_shape in [[2048], [1024]]:
                    if self.is_config_valid(w_shape, bias_shape) is False:
                        continue
                    rand_value = (
                        0.001 * paddle.rand(shape=w_shape, dtype=dtype).numpy()
                    )
                    with paddle.pir_utils.IrGuard():
                        start_prog = paddle.static.Program()
                        main_prog = paddle.static.Program()
                        with paddle.pir.core.program_guard(
                            main_prog, start_prog
                        ):
                            x = paddle.static.data(
                                name='x', shape=[3, 128, 4096], dtype=dtype
                            )

                            w = create_parameter(
                                shape=w_shape,
                                dtype=dtype,
                                initializer=paddle.nn.initializer.Assign(
                                    rand_value
                                ),
                            )
                            bias = paddle.static.data(
                                name="bias",
                                shape=bias_shape,
                                dtype=dtype,
                            )
                            res1 = paddle.matmul(x=x, y=w)
                            out = paddle.add(res1, bias)
                            out = paddle.assign(out)
                            self.feeds = {
                                "x": 0.01
                                * np.random.random((3, 128, 4096)).astype(
                                    dtype
                                ),
                                "bias": 0.01
                                * np.random.random(bias_shape).astype(dtype),
                            }
                            self.fetch_list = [out]
                            self.get_valid_op_map(dtype, w_shape)
                            yield [main_prog, start_prog], False

    def test_check_output(self):
        self.check_pass_correct(1e-3, 1e-3)


@unittest.skipIf(
    not core.is_compiled_with_cuda() or get_cuda_version() < 11020,
    "weight_only_linear requires CUDA >= 11.2",
)
class TestFusedWeightOnlyLinearPass_NoBias(PassTest):
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
            }
        elif dtype == "float16":
            self.valid_op_map = {
                "pd_op.weight_only_linear": 1,
                "pd_op.weight_quantize": 1,
                "pd_op.matmul": 0,
            }

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))
        self.pass_attr_list = [{'fused_weight_only_linear_pass': {}}]

    def sample_program(self):
        for dtype in ['float16', "float32"]:
            for w_shape in [[4096, 2048], [4096, 1024]]:
                rand_value = (
                    0.001 * paddle.rand(shape=w_shape, dtype=dtype).numpy()
                )
                with paddle.pir_utils.IrGuard():
                    start_prog = paddle.static.Program()
                    main_prog = paddle.static.Program()
                    with paddle.pir.core.program_guard(main_prog, start_prog):
                        x = paddle.static.data(
                            name='x', shape=[3, 128, 4096], dtype=dtype
                        )

                        w = create_parameter(
                            shape=w_shape,
                            dtype=dtype,
                            initializer=paddle.nn.initializer.Assign(
                                rand_value
                            ),
                        )

                        out = paddle.matmul(x=x, y=w)
                        out = paddle.assign(out)
                        self.feeds = {
                            "x": 0.01
                            * np.random.random((3, 128, 4096)).astype(dtype),
                        }
                        self.fetch_list = [out]
                        self.get_valid_op_map(dtype, w_shape)
                        yield [main_prog, start_prog], False

    def test_check_output(self):
        self.check_pass_correct(1e-3, 1e-3)


@unittest.skipIf(
    not core.is_compiled_with_cuda() or get_cuda_version() < 11020,
    "weight_only_linear requires CUDA >= 11.2",
)
class TestFusedWeightOnlyLinearPass_Weight_Only_Int8(
    TestFusedWeightOnlyLinearPass_NoBias
):
    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))
        self.pass_attr_list = [
            {
                'fused_weight_only_linear_pass': {
                    "weight_only_algo": "weight_only_int8"
                }
            }
        ]


@unittest.skipIf(
    not core.is_compiled_with_cuda() or get_cuda_version() < 11020,
    "weight_only_linear requires CUDA >= 11.2",
)
class TestFusedWeightOnlyLinearPass_Weight_Only_Int8_WithBias(
    TestFusedWeightOnlyLinearPass_WithBias
):
    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))
        self.pass_attr_list = [
            {
                'fused_weight_only_linear_pass': {
                    "weight_only_algo": "weight_only_int8",
                }
            }
        ]


if __name__ == "__main__":
    unittest.main()
