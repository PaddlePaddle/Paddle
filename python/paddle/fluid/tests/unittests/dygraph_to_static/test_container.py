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

import os
<<<<<<< HEAD
import tempfile
import unittest

import numpy as np

import paddle


class BufferLayers(paddle.nn.Layer):
    def __init__(self, out_channel):
        super().__init__()
=======
import paddle
import unittest
import numpy as np
import tempfile


class BufferLayers(paddle.nn.Layer):

    def __init__(self, out_channel):
        super(BufferLayers, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.out_channel = out_channel

    def forward(self, x):
        mean = paddle.mean(x)
<<<<<<< HEAD
        if mean < 0.0:
=======
        if mean < 0.:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            x = x * self._mask()

        out = x - mean
        return out

    def _mask(self):
        return paddle.to_tensor(np.zeros([self.out_channel], 'float32'))


class SequentialNet(paddle.nn.Layer):
<<<<<<< HEAD
    def __init__(self, sub_layer, in_channel, out_channel):
        super().__init__()
        self.layer = paddle.nn.Sequential(
            ('l1', paddle.nn.Linear(in_channel, in_channel)),
            ('l2', paddle.nn.Linear(in_channel, out_channel)),
            ('l3', sub_layer(out_channel)),
        )
=======

    def __init__(self, sub_layer, in_channel, out_channel):
        super(SequentialNet, self).__init__()
        self.layer = paddle.nn.Sequential(
            ('l1', paddle.nn.Linear(in_channel, in_channel)),
            ('l2', paddle.nn.Linear(in_channel, out_channel)),
            ('l3', sub_layer(out_channel)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def forward(self, x):
        out = self.layer(x)
        return out


class NestSequentialNet(paddle.nn.Layer):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self):
        super().__init__()
        group1 = paddle.nn.Sequential(
            paddle.nn.Linear(10, 10),
            paddle.nn.Sigmoid(),
        )
        group2 = paddle.nn.Sequential(
            paddle.nn.Linear(10, 3),
            paddle.nn.ReLU(),
        )
        self.layers = paddle.nn.Sequential(group1, group2)

    def forward(self, x):
        return self.layers(x)


class TestSequential(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        paddle.set_device('cpu')
        self.seed = 2021
        self.temp_dir = tempfile.TemporaryDirectory()
        self._init_config()

    def _init_config(self):
        self.net = SequentialNet(BufferLayers, 10, 3)
        self.model_path = os.path.join(self.temp_dir.name, 'sequential_net')

    def tearDown(self):
        self.temp_dir.cleanup()

    def _init_seed(self):
        paddle.seed(self.seed)
        np.random.seed(self.seed)

    def _run(self, to_static):
        self._init_seed()
        if to_static:
            self.net = paddle.jit.to_static(self.net)
        x = paddle.rand([16, 10], 'float32')
        out = self.net(x)
        if to_static:
            load_out = self._test_load(self.net, x)
            np.testing.assert_allclose(
                load_out,
                out,
                rtol=1e-05,
<<<<<<< HEAD
                err_msg='load_out is {}\\st_out is {}'.format(load_out, out),
            )
=======
                err_msg='load_out is {}\\st_out is {}'.format(load_out, out))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        return out

    def test_train(self):
        paddle.jit.set_code_level(100)
        dy_out = self._run(to_static=False)
        st_out = self._run(to_static=True)
        np.testing.assert_allclose(
            dy_out,
            st_out,
            rtol=1e-05,
            err_msg='dygraph_res is {}\nstatic_res is {}'.format(
<<<<<<< HEAD
                dy_out, st_out
            ),
        )
=======
                dy_out, st_out))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def _test_load(self, net, x):
        paddle.jit.save(net, self.model_path)
        load_net = paddle.jit.load(self.model_path)
        out = load_net(x)
        return out


class TestNestSequential(TestSequential):
<<<<<<< HEAD
    def _init_config(self):
        self.net = NestSequentialNet()
        self.model_path = os.path.join(
            self.temp_dir.name, 'nested_sequential_net'
        )
=======

    def _init_config(self):
        self.net = NestSequentialNet()
        self.model_path = os.path.join(self.temp_dir.name,
                                       'nested_sequential_net')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


if __name__ == '__main__':
    unittest.main()
