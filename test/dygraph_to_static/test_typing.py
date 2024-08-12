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
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
from dygraph_to_static_utils import Dy2StTestBase, test_legacy_and_pir

import paddle


class BaseLayer(paddle.nn.Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self._linear = paddle.nn.Linear(in_size, out_size)

    def build(self, x):
        out1 = self._linear(x)
        out2 = paddle.mean(out1)
        return out1, out2


class LinearNetWithTuple(BaseLayer):
    def __init__(self, in_size, out_size):
        super().__init__(in_size, out_size)

    def forward(self, x) -> tuple[paddle.Tensor, str]:
        out1, out2 = self.build(x)
        return (out2, 'str')


class LinearNetWithTuple2(BaseLayer):
    def __init__(self, in_size, out_size):
        super().__init__(in_size, out_size)

    def forward(self, x) -> tuple[paddle.Tensor, np.array]:
        out1, out2 = self.build(x)
        return (out2, np.ones([4, 16]))


class LinearNetWithList(BaseLayer):
    def __init__(self, in_size, out_size):
        super().__init__(in_size, out_size)

    def forward(self, x) -> list[paddle.Tensor]:
        out1, out2 = self.build(x)
        return [out2]


class LinearNetWithDict(BaseLayer):
    def __init__(self, in_size, out_size):
        super().__init__(in_size, out_size)

    def forward(self, x) -> dict[str, paddle.Tensor]:
        out1, out2 = self.build(x)
        return {'out': out2}


class TestTyping(Dy2StTestBase):
    def setUp(self):
        self.in_num = 16
        self.out_num = 16
        self.x = paddle.randn([4, 16])
        self.spec = [paddle.static.InputSpec(shape=[None, 16], dtype='float32')]

        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def build_net(self):
        return LinearNetWithTuple(self.in_num, self.out_num)

    def save_and_load(self, suffix=''):
        path = os.path.join(self.temp_dir.name, 'layer_typing_' + suffix)
        paddle.jit.save(self.net, path, input_spec=self.spec)
        return paddle.jit.load(path)

    def run_dy(self):
        out, _ = self.net(self.x)
        return out

    @test_legacy_and_pir
    def test_type(self):
        self.net = self.build_net()
        out = self.run_dy()
        load_net = self.save_and_load('tuple')
        load_out = load_net(self.x)
        np.testing.assert_allclose(out, load_out, rtol=1e-05)


class TestTypingTuple(TestTyping):
    def build_net(self):
        return LinearNetWithTuple2(self.in_num, self.out_num)

    def run_dy(self):
        out, np_data = self.net(self.x)
        self.assertTrue(np.equal(np_data, np.ones_like(np_data)).all())
        return out


class TestTypingList(TestTyping):
    def build_net(self):
        return LinearNetWithList(self.in_num, self.out_num)

    def run_dy(self):
        out = self.net(self.x)[0]
        return out


class TestTypingDict(TestTyping):
    def build_net(self):
        return LinearNetWithDict(self.in_num, self.out_num)

    def run_dy(self):
        out = self.net(self.x)['out']
        return out


if __name__ == '__main__':
    unittest.main()
