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

import paddle
import unittest
import numpy as np


class BufferLayers(paddle.nn.Layer):
    def __init__(self, out_channel):
        super(BufferLayers, self).__init__()
        self.out_channel = out_channel

    def forward(self, x):
        mean = paddle.mean(x)
        if mean < 0.:
            x = x * self._mask()

        out = x - mean
        return out

    def _mask(self):
        return paddle.to_tensor(np.zeros([self.out_channel], 'float32'))


class SequentialNet(paddle.nn.Layer):
    def __init__(self, sub_layer, in_channel, out_channel):
        super(SequentialNet, self).__init__()
        self.layer = paddle.nn.Sequential(
            ('l1', paddle.nn.Linear(in_channel, in_channel)),
            ('l2', paddle.nn.Linear(in_channel, out_channel)),
            ('l3', sub_layer(out_channel)))

    def forward(self, x):
        out = self.layer(x)
        return out


class TestSequential(unittest.TestCase):
    def setUp(self):
        paddle.set_device('cpu')
        self.seed = 2021

    def _init_seed(self):
        paddle.seed(self.seed)
        np.random.seed(self.seed)

    def _run(self, to_static):
        self._init_seed()
        net = SequentialNet(BufferLayers, 10, 3)
        if to_static:
            net = paddle.jit.to_static(net)
        x = paddle.rand([16, 10], 'float32')
        out = net(x)
        if to_static:
            load_out = self._test_load(net, x)
            self.assertTrue(
                np.allclose(load_out, out),
                msg='load_out is {}\st_out is {}'.format(load_out, out))

        return out

    def test_train(self):
        paddle.jit.set_code_level(100)
        dy_out = self._run(to_static=False)
        st_out = self._run(to_static=True)
        self.assertTrue(
            np.allclose(dy_out, st_out),
            msg='dygraph_res is {}\nstatic_res is {}'.format(dy_out, st_out))

    def _test_load(self, net, x):
        model_path = './sequential_net'
        paddle.jit.save(net, model_path)
        load_net = paddle.jit.load(model_path)
        out = load_net(x)
        return out


if __name__ == '__main__':
    unittest.main()
