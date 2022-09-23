# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import os
import numpy as np
import tempfile


def forward_post_hook1(layer, input, output):
    return output + output


def forward_pre_hook1(layer, input):
    input_return = (input[0] * 2, )
    return input_return


class SimpleNet(paddle.nn.Layer):

    def __init__(self, ):
        super(SimpleNet, self).__init__()
        self.fc1 = paddle.nn.Linear(10, 10)
        # sublayer1 register post hook
        self.fc1.register_forward_post_hook(forward_post_hook1)

        self.fc2 = paddle.nn.Linear(10, 10)
        # sublayer2 register pre hook
        self.fc2.register_forward_pre_hook(forward_pre_hook1)

        # register pre/post hook
        self.register_forward_pre_hook(forward_pre_hook1)
        self.register_forward_post_hook(forward_post_hook1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        out = paddle.mean(x)

        return out


class TestNestLayerHook(unittest.TestCase):

    def setUp(self):
        paddle.seed(2022)
        self.x = paddle.randn([4, 10])
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.temp_dir.name, 'net_hook')

    def tearDown(self):
        self.temp_dir.cleanup()

    def train_net(self, to_static=False):
        paddle.seed(2022)
        net = SimpleNet()
        if to_static:
            net = paddle.jit.to_static(net)
        out = net(self.x)

        if to_static:
            paddle.jit.save(net, self.path)

        return out.numpy()[0]

    def load_train(self):
        net = paddle.jit.load(self.path)
        out = net(self.x)
        return out.numpy()[0]

    def test_hook(self):
        dy_out = self.train_net(to_static=False)
        st_out = self.train_net(to_static=True)
        load_out = self.load_train()
        print(st_out, dy_out, load_out)
        np.testing.assert_allclose(
            st_out,
            dy_out,
            rtol=1e-05,
            err_msg='dygraph_res is {}\nstatic_res is {}'.format(
                dy_out, st_out))
        np.testing.assert_allclose(
            st_out,
            load_out,
            rtol=1e-05,
            err_msg='load_out is {}\nstatic_res is {}'.format(load_out, st_out))


if __name__ == "__main__":
    unittest.main()
