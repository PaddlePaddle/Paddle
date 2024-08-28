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
        num_layers = 8
        x_shape = [bsz, seq_len, num_head, head_dim]
        weight_shape = [dim, dim]

        with paddle.pir_utils.IrGuard():
            start_prog = paddle.static.Program()
            main_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(name="x", shape=x_shape, dtype="float16")
                w_vec = []
                res_vec = []
                for i in range(num_layers):
                    w_vec.append(
                        paddle.static.data(
                            name=f"w{i}", shape=weight_shape, dtype="float16"
                        )
                    )

                x = paddle.reshape(x, [bsz, seq_len, dim])
                for i in range(num_layers):
                    res_vec.append(paddle.matmul(x, w_vec[i]))

                for i in range(num_layers):
                    res_vec[i] = paddle.assign(res_vec[i])

                self.pass_attr_list = [{"horizontal_fuse_pass": {}}]
                self.feeds = {
                    "x": np.random.random(x_shape).astype("float16"),
                }
                for i in range(num_layers):
                    self.feeds[f"w{i}"] = np.random.random(weight_shape).astype(
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
                x = paddle.static.data(name="x", shape=x_shape, dtype="float16")
                weight_vec = []
                bias_vec = []
                res_vec = []
                for i in range(num_layers):
                    weight_vec.append(
                        paddle.static.data(
                            name=f"w{i}", shape=weight_shape, dtype="float16"
                        )
                    )
                    bias_vec.append(
                        paddle.static.data(
                            name=f"b{i}", shape=bias_shape, dtype="float16"
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
                    {"matmul_add_act_fuse_pass": {"use_cutlass": use_cutlass}},
                    {"horizontal_fuse_pass": {}},
                ]
                self.feeds = {
                    "x": np.random.random(x_shape).astype("float16"),
                }
                for i in range(num_layers):
                    self.feeds[f"w{i}"] = np.random.random(weight_shape).astype(
                        "float16"
                    )
                    self.feeds[f"b{i}"] = np.random.random(bias_shape).astype(
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


class TestMatmulHorizontalFusePatternFirstCase(PassTest):
    """
                 x
                 |
       -----------------------
      |       |       |       |
    matmul  matmul  multiply  matmul
      |       |       |       |
      res1    res2    res3    res4
    """

    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            start_prog = paddle.static.Program()
            main_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                res_vec = []

                # Define shapes for input and weights
                x_shape = [20, 64, 56]
                w1_shape = [56, 68]
                w2_shape = [64, 56]
                w3_shape = [56, 78]
                w4_shape = [56, 98]

                # Define input and weight tensors
                x = paddle.static.data(name="x", shape=x_shape, dtype="float16")
                w1 = paddle.static.data(
                    name="w1", shape=w1_shape, dtype="float16"
                )
                w2 = paddle.static.data(
                    name="w2", shape=w2_shape, dtype="float16"
                )
                w3 = paddle.static.data(
                    name="w3", shape=w3_shape, dtype="float16"
                )
                w4 = paddle.static.data(
                    name="w4", shape=w4_shape, dtype="float16"
                )

                # Reshape input tensor
                x = paddle.reshape(x, x_shape)

                # Perform matmul and multiply operations
                res_vec.append(paddle.matmul(x, w1))
                res_vec.append(paddle.multiply(x, w2))
                res_vec.append(paddle.matmul(x, w3))
                res_vec.append(paddle.matmul(x, w4))

                # Assign results to the program
                for result in res_vec:
                    paddle.assign(result)

                # Define pass attributes
                self.pass_attr_list = [{"horizontal_fuse_pass": {}}]

                # Define input feed values
                self.feeds = {
                    "x": np.random.random(x_shape).astype("float16"),
                    "w1": np.random.random(w1_shape).astype("float16"),
                    "w2": np.random.random(w2_shape).astype("float16"),
                    "w3": np.random.random(w3_shape).astype("float16"),
                    "w4": np.random.random(w4_shape).astype("float16"),
                }

                # Define fetch list
                self.fetch_list = res_vec

                # Define valid operation map
                self.valid_op_map = {
                    "pd_op.concat": 1,
                    "pd_op.matmul": 1,
                    "pd_op.split": 1,
                    "pd_op.multiply": 1,
                }

                yield [main_prog, start_prog], False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct(atol=2e-3, rtol=2e-3)


class TestMatmulHorizontalFusePatternSecondCase(PassTest):
    """matmul's input is Y"""

    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            start_prog = paddle.static.Program()
            main_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                res_vec = []

                # Define shapes for input and weights
                x_shape = [20, 64, 56]
                w1_shape = [56, 64]
                w2_shape = [65, 64]

                # Define input and weight tensors
                x = paddle.static.data(name="x", shape=x_shape, dtype="float16")
                w1 = paddle.static.data(
                    name="w1", shape=w1_shape, dtype="float16"
                )
                w2 = paddle.static.data(
                    name="w2", shape=w2_shape, dtype="float16"
                )
                w3 = paddle.static.data(
                    name="w3", shape=w2_shape, dtype="float16"
                )

                weights = [w1, w2, w3]

                # Reshape input tensor
                x = paddle.reshape(x, x_shape)

                # Perform matmul operations
                for w in weights:
                    res_vec.append(paddle.matmul(w, x))

                # Define pass attributes
                self.pass_attr_list = [{"horizontal_fuse_pass": {}}]

                # Define input feed values
                self.feeds = {
                    "x": np.random.random(x_shape).astype("float16"),
                    "w1": np.random.random(w1_shape).astype("float16"),
                    "w2": np.random.random(w2_shape).astype("float16"),
                    "w3": np.random.random(w2_shape).astype("float16"),
                }

                # Define fetch list
                self.fetch_list = res_vec

                # Define valid operation map
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


class TestMatmulHorizontalFusePatternBadCase(PassTest):
    r"""
    matmul's weight not a ParameterOp\ConstantTensorOp\DataOp
    """

    def is_program_valid(self, program=None):
        return True

    def sample_program(self):

        with paddle.pir_utils.IrGuard():
            start_prog = paddle.static.Program()
            main_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                res_vec = []
                x_shape = [20, 64, 56]
                w_shape = [56, 88]

                x = paddle.static.data(name="x", shape=x_shape, dtype="float16")

                input_1 = paddle.static.data(
                    name="input_1", shape=w_shape, dtype="float16"
                )
                input_2 = paddle.static.data(
                    name="input_2", shape=w_shape, dtype="float16"
                )
                x = paddle.reshape(x, x_shape)

                w1 = paddle.multiply(input_1, input_2)
                w2 = paddle.multiply(input_1, input_2)
                w3 = paddle.multiply(input_1, input_2)

                res_vec.append(paddle.matmul(x, w1))
                res_vec.append(paddle.matmul(x, w2))
                res_vec.append(paddle.matmul(x, w3))

                for one in res_vec:
                    paddle.assign(one)

                self.pass_attr_list = [
                    {"horizontal_fuse_pass": {}},
                ]

                self.feeds = {
                    "x": np.random.random(x_shape).astype("float16"),
                    "input_1": np.random.random(w_shape).astype("float16"),
                    "input_2": np.random.random(w_shape).astype("float16"),
                }
                self.fetch_list = res_vec
                self.valid_op_map = {
                    "pd_op.concat": 0,
                    "pd_op.matmul": 3,
                    "pd_op.split": 0,
                    "pd_op.multiply": 3,
                }
                yield [main_prog, start_prog], False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct(atol=2e-3, rtol=2e-3)


if __name__ == "__main__":
    unittest.main()
