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
import unittest

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle.base import core
from paddle.incubate.autograd import primapi


class TestPrimFlags(unittest.TestCase):
    def test_prim_flags(self):
        self.assertFalse(core._is_bwd_prim_enabled())
        self.assertFalse(core._is_fwd_prim_enabled())

        os.environ['FLAGS_prim_backward'] = "True"
        core.check_and_set_prim_all_enabled()
        self.assertTrue(core._is_bwd_prim_enabled())
        os.environ['FLAGS_prim_forward'] = "True"
        core.check_and_set_prim_all_enabled()
        self.assertTrue(core._is_fwd_prim_enabled())
        os.environ['FLAGS_prim_all'] = "False"
        core.check_and_set_prim_all_enabled()
        self.assertFalse(core._is_bwd_prim_enabled())
        self.assertFalse(core._is_fwd_prim_enabled())

        os.environ['FLAGS_prim_all'] = "True"
        core.check_and_set_prim_all_enabled()
        self.assertTrue(core._is_bwd_prim_enabled())
        self.assertTrue(core._is_fwd_prim_enabled())

        del os.environ['FLAGS_prim_all']
        os.environ['FLAGS_prim_backward'] = "False"
        core.check_and_set_prim_all_enabled()
        self.assertFalse(core._is_bwd_prim_enabled())
        os.environ['FLAGS_prim_forward'] = "False"
        core.check_and_set_prim_all_enabled()
        self.assertFalse(core._is_fwd_prim_enabled())

        del os.environ['FLAGS_prim_backward']
        core.check_and_set_prim_all_enabled()
        self.assertFalse(core._is_bwd_prim_enabled())
        del os.environ['FLAGS_prim_forward']
        core.check_and_set_prim_all_enabled()
        self.assertFalse(core._is_fwd_prim_enabled())

        core.set_prim_eager_enabled(True)
        self.assertTrue(core._is_eager_prim_enabled())

        with self.assertRaises(TypeError):
            core._test_use_sync("aaaa")

        core._set_prim_all_enabled(True)
        self.assertTrue(core._is_all_prim_enabled())

        core._set_prim_all_enabled(False)
        self.assertFalse(core._is_all_prim_enabled())


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
            y = F.softmax(x)
            blocks = main_program.blocks

            fwd_ops = [op.type for op in blocks[0].ops]
            # Ensure that softmax in original block
            self.assertTrue('softmax' in fwd_ops)

            primapi.to_prim(blocks)

            fwd_ops_new = [op.type for op in blocks[0].ops]
            # Ensure that softmax is splitted into small ops
            self.assertTrue('softmax' not in fwd_ops_new)

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
            y = F.softmax(x)
            blocks = main_program.blocks

            fwd_ops = [op.type for op in blocks[0].ops]
            # Ensure that softmax in original block
            self.assertTrue('softmax' in fwd_ops)

            primapi.to_prim(blocks)

            fwd_ops_new = [op.type for op in blocks[0].ops]
            # Ensure that softmax is splitted into small ops
            self.assertTrue('softmax' in fwd_ops_new)

        exe = paddle.static.Executor()
        exe.run(startup_program)
        _ = exe.run(main_program, feed={'x': inputs}, fetch_list=[y])
        paddle.disable_static()
        core._set_prim_forward_enabled(False)

    def test_prim_forward_blacklist(self):
        self.not_in_blacklist()
        core._set_prim_forward_blacklist("softmax")
        self.in_blacklist()


class PrimeNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x1 = F.softmax(x)
        x2 = paddle.exp(x1)
        res = paddle.nn.functional.relu(x2)
        return res


class TestPrimBackwardBlacklistFlags(unittest.TestCase):
    def train(self):
        x = paddle.randn([2, 4])
        x.stop_gradient = False
        net = PrimeNet()
        net = paddle.jit.to_static(net, full_graph=True)

        out = net(x)
        loss = paddle.mean(out)
        loss.backward()
        self.check_prim(net)

    def check_prim(self, net):
        block = net.forward.program_cache.last()[-1][-1].train_program.block
        ops = [op.type for op in block(0).ops]
        self.assertTrue('softmax_grad' in ops)
        self.assertTrue('exp_grad' in ops)
        self.assertTrue('relu_grad' not in ops)

    def test_prim_backward_blacklist(self):
        core._set_prim_all_enabled(True)
        core._set_prim_backward_blacklist("softmax", "exp")
        self.train()
        core._set_prim_all_enabled(False)


if __name__ == '__main__':
    unittest.main()
