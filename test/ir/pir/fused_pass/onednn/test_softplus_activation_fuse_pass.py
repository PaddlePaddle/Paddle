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

paddle.enable_static()

activation_type = [
    "abs",
    "gelu",
    "hard_sigmoid",
    "hard_swish",
    "leaky_relu",
    "mish",
    "relu",
    "relu6",
    "sigmoid",
    "sqrt",
    "swish",
    "tanh",
]


class TestSoftplusActivationFusePattern(PassTest):
    r"""
       x
       |
    softplus
       |
      act
       |
      out
    """

    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        x_shape = [3, 2]
        for act_op in activation_type:
            with paddle.pir_utils.IrGuard():
                start_prog = paddle.static.Program()
                main_prog = paddle.static.Program()
                with paddle.pir.core.program_guard(main_prog, start_prog):
                    x = paddle.static.data(
                        name='x', shape=x_shape, dtype='float32'
                    )

                    softplus_out = paddle.nn.functional.softplus(x)

                    if act_op == "abs":
                        out = paddle.abs(softplus_out)
                    elif act_op == "gelu":
                        out = paddle.nn.functional.gelu(softplus_out)
                    elif act_op == "hard_sigmoid":
                        out = paddle.nn.functional.hardsigmoid(softplus_out)
                    elif act_op == "hard_swish":
                        out = paddle.nn.functional.hardswish(softplus_out)
                    elif act_op == "leaky_relu":
                        out = paddle.nn.functional.leaky_relu(softplus_out)
                    elif act_op == "mish":
                        out = paddle.nn.functional.mish(softplus_out)
                    elif act_op == "relu":
                        out = paddle.nn.functional.relu(softplus_out)
                    elif act_op == "relu6":
                        out = paddle.nn.functional.relu6(softplus_out)
                    elif act_op == "sigmoid":
                        out = paddle.nn.functional.sigmoid(softplus_out)
                    elif act_op == "sqrt":
                        out = paddle.sqrt(softplus_out)
                    elif act_op == "swish":
                        out = paddle.nn.functional.swish(softplus_out)
                    elif act_op == "tanh":
                        out = paddle.nn.functional.tanh(softplus_out)

                    out = paddle.assign(out)
                    self.pass_attr_list = [
                        {"softplus_activation_fuse_pass": {}}
                    ]
                    self.feeds = {
                        "x": np.random.random(x_shape).astype("float32")
                    }
                    self.fetch_list = [out]
                    self.valid_op_map = {
                        "onednn_op.fused_softplus": 1,
                        "pd_op.matmul": 0,
                        "pd_op.add": 0,
                        "pd_op.abs": 0,
                        "pd_op.gelu": 0,
                        "pd_op.hard_sigmoid": 0,
                        "pd_op.hard_swish": 0,
                        "pd_op.leaky_relu": 0,
                        "pd_op.mish": 0,
                        "pd_op.relu": 0,
                        "pd_op.relu6": 0,
                        "pd_op.sigmoid": 0,
                        "pd_op.sqrt": 0,
                        "pd_op.swish": 0,
                        "pd_op.tanh": 0,
                    }

                    yield [main_prog, start_prog], False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestSoftplusGeluTanhFusePattern(PassTest):
    r"""
       x
       |
    softplus
       |
    gelu_tanh
       |
      out
    """

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=[3, 2], dtype='float32')
                softplus_out = paddle.nn.functional.softplus(x)
                out = paddle.nn.functional.gelu(softplus_out, approximate=True)
                out = paddle.assign(out)
                self.pass_attr_list = [{'softplus_activation_fuse_pass': {}}]
                self.feeds = {"x": np.random.random((3, 2)).astype("float32")}
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fused_softplus": 1,
                    "pd_op.gelu": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestSoftplusClipFusePattern(PassTest):
    r"""
       x
       |
    softplus
       |
     clip
       |
      out
    """

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=[3, 2], dtype='float32')
                softplus_out = paddle.nn.functional.softplus(x)
                out = paddle.clip(softplus_out)
                out = paddle.assign(out)
                self.pass_attr_list = [{'softplus_activation_fuse_pass': {}}]
                self.feeds = {"x": np.random.random((3, 2)).astype("float32")}
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fused_softplus": 1,
                    "pd_op.clip": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


if __name__ == "__main__":
    unittest.main()
