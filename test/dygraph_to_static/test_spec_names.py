# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from dygraph_to_static_utils import (
    Dy2StTestBase,
    test_ast_only,
)

import paddle
from paddle.nn import Layer


class Net(Layer):
    def __init__(self):
        super().__init__()
        self.fc = paddle.nn.Linear(16, 3)

    def forward(self, x, y, m, n):
        inputs = [x, y, m, n]
        outs = []
        for var in inputs:
            out = paddle.reshape(x, [-1, 16])
            out = self.fc(out)
            outs.append(out)

        out = paddle.stack(outs)
        return paddle.sum(out)


class TestArgsSpecName(Dy2StTestBase):
    def read_from_dataset(self):
        self.x = paddle.randn([4, 2, 8])
        self.y = paddle.randn([4, 2, 8])
        self.m = paddle.randn([4, 2, 8])
        self.n = paddle.randn([4, 2, 8])

    @test_ast_only
    def test_spec_name_hash(self):
        net = Net()
        net = paddle.jit.to_static(net)

        # Convert into program with four input
        self.read_from_dataset()
        self.run_test(net, [self.x, self.y, self.m, self.n], 1, [0, 1, 2, 3])

        # Convert into program with three input
        self.read_from_dataset()
        self.run_test(net, [self.x, self.x, self.m, self.n], 1, [0, 0, 1, 2])

        # Convert into program with two input
        self.read_from_dataset()
        self.run_test(net, [self.x, self.x, self.m, self.m], 1, [0, 0, 1, 1])

        # Use Cache Program
        self.read_from_dataset()
        self.run_test(net, [self.n, self.n, self.y, self.y], 1, [0, 0, 1, 1])

        # Convert into program with two input
        self.read_from_dataset()
        self.run_test(net, [self.x, self.y, self.x, self.y], 1, [0, 1, 0, 1])

        # Use Cache Program
        self.read_from_dataset()
        self.run_test(net, [self.m, self.n, self.m, self.n], 1, [0, 1, 0, 1])

        # Convert into program with one input
        self.read_from_dataset()
        self.run_test(net, [self.x, self.x, self.x, self.x], 1, [0, 0, 0, 0])

        # Use Cache Program
        self.read_from_dataset()
        self.run_test(net, [self.m, self.m, self.m, self.m], 1, [0, 0, 0, 0])

    def run_test(self, net, inputs, trace_count, mode):
        out = net(*inputs)
        self.assertEqual(net.forward.get_traced_count(), trace_count)


if __name__ == '__main__':
    unittest.main()
