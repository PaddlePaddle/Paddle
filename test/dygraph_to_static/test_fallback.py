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
from dygraph_to_static_util import ast_only_test

import paddle


def support_func(x):
    return 2 * x


def unsupport_func(x):
    x = 2 * x
    t = x.numpy()
    t = np.ones(t)
    return paddle.to_tensor(t)


class SuppportNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return support_func(x)


class UnsuppportNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if self.training:
            return unsupport_func(x)
        else:
            return unsupport_func(x - 1)


class TestFallback(unittest.TestCase):
    def setUp(self):
        self.x = paddle.to_tensor([2]).astype('int')

    def tearDown(self):
        pass

    def test_case_support(self):
        output = paddle.jit.to_static(support_func)(self.x)
        np.testing.assert_allclose(output.numpy(), 4)

    def test_case_func_fallback(self):
        build_strategy = paddle.static.BuildStrategy()
        build_strategy.build_cinn_pass = True
        output = paddle.jit.to_static(
            unsupport_func, build_strategy=build_strategy
        )(self.x)
        np.testing.assert_allclose(output.numpy(), unsupport_func(self.x))

    def test_case_net_fallback(self):
        s_net = SuppportNet()
        u_net = UnsuppportNet()
        np.testing.assert_allclose(
            paddle.jit.to_static(s_net)(self.x).numpy(), 4
        )
        build_strategy = paddle.static.BuildStrategy()
        build_strategy.build_cinn_pass = True
        np.testing.assert_allclose(
            paddle.jit.to_static(u_net, build_strategy=build_strategy)(
                self.x
            ).numpy(),
            u_net(self.x).numpy(),
        )

    @ast_only_test
    def test_case_net_error(self):
        s_net = SuppportNet()
        u_net = UnsuppportNet()
        np.testing.assert_allclose(
            paddle.jit.to_static(s_net)(self.x).numpy(), 4
        )
        build_strategy = paddle.static.BuildStrategy()
        build_strategy.build_cinn_pass = False
        with self.assertRaises(TypeError):
            np.testing.assert_allclose(
                paddle.jit.to_static(u_net, build_strategy=build_strategy)(
                    self.x
                ).numpy(),
                u_net(self.x).numpy(),
            )

    def test_case_training(self):
        build_strategy = paddle.static.BuildStrategy()
        build_strategy.build_cinn_pass = True
        u_net = paddle.jit.to_static(
            UnsuppportNet(), build_strategy=build_strategy
        )
        u_net.eval()
        np.testing.assert_allclose(u_net(self.x).numpy(), [1, 1])
        assert u_net.training is False, "Training must be false."

    def test_case_save_error(self):
        """
        test the save will raise error.
        """
        u_net = UnsuppportNet()
        u_net = paddle.jit.to_static(
            u_net, input_spec=[paddle.static.InputSpec(name='x', shape=[1])]
        )
        with self.assertRaises(TypeError):
            paddle.jit.save(u_net, path="model")

    def test_case_save_error_2(self):
        """
        test the save will raise error.
        """
        u_net = UnsuppportNet()
        build_strategy = paddle.static.BuildStrategy()
        build_strategy.build_cinn_pass = True
        u_net = paddle.jit.to_static(u_net, build_strategy=build_strategy)
        u_net(self.x)
        with self.assertRaises(RuntimeError):
            print(u_net.forward.main_program)

    def test_case_flag(self):
        """
        test the flags is working. TODO: add a global flags.
        """
        pass


if __name__ == "__main__":
    unittest.main()
