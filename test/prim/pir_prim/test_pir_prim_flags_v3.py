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

import paddle
from paddle.decomposition import decomp


class PrimeNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x1 = paddle.tanh(x)
        x2 = paddle.exp(x)
        res = paddle.matmul(x1, x2)
        return res


class TestPrimMatmulDefaultRevert(unittest.TestCase):
    def train(self):
        x = paddle.randn([4, 4])
        x.stop_gradient = False
        net = PrimeNet()
        net.forward = paddle.jit.to_static(full_graph=True)(net.forward)
        out = net(x)
        loss = paddle.mean(out)
        loss.backward()
        self.check_prim(net)

    def check_prim(self, net):
        program = net.forward.program_cache.last()[-1][-1].train_program
        if isinstance(
            program, paddle.jit.dy2static.pir_partial_program.RunnableProgram
        ):
            program = program.program
        block = program.global_block()
        ops = [op.name() for op in block.ops]
        self.assertTrue('pd_op.matmul_grad' not in ops)

    def test_prim_matmul_default(self):
        with decomp.prim_guard():
            self.train()


if __name__ == '__main__':
    unittest.main()
