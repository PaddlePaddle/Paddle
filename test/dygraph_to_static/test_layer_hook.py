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

import os
import tempfile
import unittest

import numpy as np
from dygraph_to_static_utils import (
    Dy2StTestBase,
)

import paddle


def forward_post_hook1(layer, input, output):
    return output + output


def forward_pre_hook1(layer, input):
    input_return = (input[0] * 2,)
    return input_return


class SimpleNet(paddle.nn.Layer):
    def __init__(
        self,
    ):
        super().__init__()
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


class TestNestLayerHook(Dy2StTestBase):
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

        paddle.jit.save(net, self.path, input_spec=[self.x])

        return float(out)

    def load_train(self):
        net = paddle.jit.load(self.path)
        out = net(self.x)
        return float(out)

    def test_hook(self):
        dy_out = self.train_net(to_static=False)
        st_out = self.train_net(to_static=True)
        np.testing.assert_allclose(
            st_out,
            dy_out,
            rtol=1e-05,
            err_msg=f'dygraph_res is {dy_out}\nstatic_res is {st_out}',
        )
        if not paddle.base.framework.use_pir_api():
            load_out = self.load_train()
            np.testing.assert_allclose(
                st_out,
                load_out,
                rtol=1e-05,
                err_msg=f'load_out is {load_out}\nstatic_res is {st_out}',
            )


if __name__ == "__main__":
    unittest.main()
