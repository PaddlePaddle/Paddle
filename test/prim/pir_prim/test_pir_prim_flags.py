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
import paddle.nn.functional as F
from paddle.base import core
from paddle.decomposition import decompose


class TestPrimBlacklistFlags(unittest.TestCase):
    def not_in_blacklist(self):
        inputs = np.random.random([2, 3, 4]).astype("float32")
        paddle.enable_static()
        core._set_prim_forward_enabled(True)
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(
                'x', shape=inputs.shape, dtype=str(inputs.dtype)
            )
            y = F.gelu(x)

            fwd_ops = [op.name() for op in main_program.global_block().ops]
            # Ensure that tanh in original block
            self.assertTrue('pd_op.gelu' in fwd_ops)

            [y] = decompose(main_program, [y])

            fwd_ops_new = [op.name() for op in main_program.global_block().ops]
            # Ensure that tanh is splitted into small ops
            self.assertTrue('pd_op.gelu' not in fwd_ops_new)

        exe = paddle.static.Executor()
        exe.run(startup_program)
        _ = exe.run(main_program, feed={'x': inputs}, fetch_list=[y])
        paddle.disable_static()
        core._set_prim_forward_enabled(False)

    def in_blacklist(self):
        inputs = np.random.random([2, 3, 4]).astype("float32")
        paddle.enable_static()
        core._set_prim_forward_enabled(True)
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(
                'x', shape=inputs.shape, dtype=str(inputs.dtype)
            )
            y = F.gelu(x)

            fwd_ops = [op.name() for op in main_program.global_block().ops]
            # Ensure that tanh in original block
            self.assertTrue('pd_op.gelu' in fwd_ops)

            _ = decompose(main_program, [y])

            fwd_ops_new = [op.name() for op in main_program.global_block().ops]
            # Ensure that tanh is splitted into small ops
            self.assertTrue('pd_op.gelu' in fwd_ops_new)

        exe = paddle.static.Executor()
        exe.run(startup_program)
        _ = exe.run(main_program, feed={'x': inputs}, fetch_list=[y])
        paddle.disable_static()
        core._set_prim_forward_enabled(False)

    def test_prim_forward_blacklist(self):
        self.not_in_blacklist()
        core._set_prim_forward_blacklist("pd_op.gelu")
        self.in_blacklist()


class PrimeNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x1 = paddle.tanh(x)
        x2 = paddle.exp(x)
        x3 = x1 + x2
        res = paddle.nn.functional.gelu(x3)
        return res


class TestPrimBackwardBlacklistFlags(unittest.TestCase):
    def train(self):
        x = paddle.randn([2, 4])
        x.stop_gradient = False
        net = PrimeNet()
        net.forward = paddle.jit.to_static(full_graph=True)(net.forward)
        out = net(x)
        loss = paddle.mean(out)
        loss.backward()
        self.check_prim(net)

    def check_prim(self, net):
        block = net.forward.program_cache.last()[-1][
            -1
        ].train_program.global_block()
        ops = [op.name() for op in block.ops]
        self.assertTrue('pd_op.tanh_grad' in ops)
        self.assertTrue('pd_op.exp_grad' in ops)
        self.assertTrue('pd_op.gelu_grad' not in ops)

    def test_prim_backward_blacklist(self):
        core._set_prim_all_enabled(True)
        core._set_prim_backward_blacklist("tanh_grad", "exp_grad")
        self.train()
        core._set_prim_all_enabled(False)


if __name__ == '__main__':
    unittest.main()
