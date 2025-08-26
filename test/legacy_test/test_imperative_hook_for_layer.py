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

import sys
import unittest

import numpy as np

sys.path.append("../deprecated/legacy_test")

import paddle

call_forward_post_hook = False
call_forward_pre_hook = False


def forward_post_hook(layer, input, output):
    global call_forward_post_hook
    call_forward_post_hook = True


def forward_pre_hook(layer, input):
    global call_forward_pre_hook
    call_forward_pre_hook = True


def forward_post_hook1(layer, input, output):
    return output * 2


def forward_pre_hook1(layer, input):
    input_return = (input[0] * 2, input[1])
    return input_return


def forward_pre_hook_with_kwargs(layer, args, kwargs):
    kwargs['x'] = kwargs['x'] * 2
    return (args, kwargs)


class SimpleNetWithKWArgs(paddle.nn.Layer):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x, y):
        z = x + y

        return z


class TestHookWithKWArgs(unittest.TestCase):
    def test_kwargs_hook(self):
        net = SimpleNetWithKWArgs()
        remove_handler = net.register_forward_pre_hook(
            forward_pre_hook_with_kwargs, with_kwargs=True
        )

        x = paddle.randn((2, 3))
        y = paddle.randn((2, 3))

        out = net(x=x, y=y)
        np.testing.assert_allclose(out.numpy(), (x * 2 + y).numpy())

        remove_handler.remove()
        out = net(x=x, y=y)
        np.testing.assert_allclose(out.numpy(), (x + y).numpy())


if __name__ == '__main__':
    unittest.main()
