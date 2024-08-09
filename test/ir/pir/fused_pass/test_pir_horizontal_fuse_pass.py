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
from pass_test import PassTest

import paddle
from paddle.base import core

paddle.enable_static()


class TestMatmulHorizontalFusePattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        bsz = 2
        seq_len = 16
        num_head = 2
        head_dim = 16
        dim = num_head * head_dim
        num_layers = 3
        x_shape = [bsz, seq_len, num_head, head_dim]
        weight_shape = [dim, dim]

        with paddle.pir_utils.IrGuard():
            start_prog = paddle.static.Program()
            main_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=x_shape, dtype='float16')
                w_vec = []
                res_vec = []
                for i in range(num_layers):
                    w_vec.append(
                        paddle.static.data(
                            name=f'w{i}', shape=weight_shape, dtype='float16'
                        )
                    )

                x = paddle.reshape(x, [bsz, seq_len, dim])
                for i in range(num_layers):
                    res_vec.append(paddle.matmul(x, w_vec[i]))

                for i in range(num_layers):
                    res_vec[i] = paddle.assign(res_vec[i])

                self.pass_attr_list = [{'horizontal_fuse_pass': {}}]
                self.feeds = {
                    "x": np.random.random(x_shape).astype("float16"),
                }
                for i in range(num_layers):
                    self.feeds[f'w{i}'] = np.random.random(weight_shape).astype(
                        "float16"
                    )

                self.fetch_list = res_vec
                self.valid_op_map = {
                    "pd_op.concat": 1,
                    "pd_op.matmul": 1,
                    "pd_op.split": 1,
                }
                yield [main_prog, start_prog], False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct(atol=2e-3, rtol=2e-3)


class TestGemmEpilogueHorizontalFusePattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        use_cutlass = False
        if use_cutlass:
            fused_op_name = "pd_op.gemm_epilogue"
        else:
            fused_op_name = "pd_op.fc"
        bsz = 2
        seq_len = 16
        num_head = 2
        head_dim = 16
        dim = num_head * head_dim
        num_layers = 4
        x_shape = [bsz, seq_len, num_head, head_dim]
        weight_shape = [dim, dim]
        bias_shape = [dim]

        with paddle.pir_utils.IrGuard():
            start_prog = paddle.static.Program()
            main_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=x_shape, dtype='float16')
                weight_vec = []
                bias_vec = []
                res_vec = []
                for i in range(num_layers):
                    weight_vec.append(
                        paddle.static.data(
                            name=f'w{i}', shape=weight_shape, dtype='float16'
                        )
                    )
                    bias_vec.append(
                        paddle.static.data(
                            name=f'b{i}', shape=bias_shape, dtype='float16'
                        )
                    )

                x = paddle.reshape(x, [bsz, seq_len, dim])
                for i in range(num_layers):
                    res_vec.append(
                        paddle.add(paddle.matmul(x, weight_vec[i]), bias_vec[i])
                    )
                    res_vec[i] = paddle.nn.functional.relu(res_vec[i])

                for i in range(num_layers):
                    res_vec[i] = paddle.assign(res_vec[i])

                self.pass_attr_list = [
                    {'matmul_add_act_fuse_pass': {"use_cutlass": use_cutlass}},
                    {'horizontal_fuse_pass': {}},
                ]
                self.feeds = {
                    "x": np.random.random(x_shape).astype("float16"),
                }
                for i in range(num_layers):
                    self.feeds[f'w{i}'] = np.random.random(weight_shape).astype(
                        "float16"
                    )
                    self.feeds[f'b{i}'] = np.random.random(bias_shape).astype(
                        "float16"
                    )

                self.fetch_list = res_vec
                self.valid_op_map = {
                    "pd_op.concat": 2,
                    fused_op_name: 1,
                    "pd_op.split": 1,
                }
                yield [main_prog, start_prog], False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct(atol=2e-3, rtol=2e-3)


if __name__ == "__main__":
    unittest.main()
