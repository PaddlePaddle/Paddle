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

import paddle
from paddle.jit.sot.symbolic.compile_cache import CompileSIRCache


class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.dropout = paddle.nn.Dropout(p=0.5)

    def forward(self, x):
        if self.training:
            out1 = paddle.nn.functional.dropout(x, p=0.5, training=True)
        else:
            out1 = paddle.nn.functional.dropout(x, p=0.5, training=False)
        out1 = self.dropout(out1)
        return out1


class TestModelSwitchTraining(unittest.TestCase):
    def setUp(self):
        self.seed = 1127
        self.net = SimpleNet()
        # singleton
        self.compile_cache = CompileSIRCache()

    def check_mode(self, is_train):
        self.assertEqual(len(self.compile_cache.cache), 1)
        mode = next(
            iter(self.compile_cache.cache.values())
        ).partial_program.training
        self.assertEqual(mode, is_train)

    def get_dygraph_out(self, input):
        paddle.seed(self.seed)
        self.net.eval()
        eval_result = self.net(input)
        self.net.train()
        train_result = self.net(input)
        return eval_result, train_result

    def get_static_out(self, input):
        paddle.seed(self.seed)
        self.compile_cache.clear()
        static_net = paddle.jit.to_static(self.net, full_graph=False)
        static_net.eval()
        eval_result = static_net(input)
        self.check_mode(is_train=False)
        self.compile_cache.clear()

        static_net.train()
        train_result = static_net(input)
        self.check_mode(is_train=True)
        return eval_result, train_result

    def test_model_switch_training(self):
        input = paddle.rand((10, 10))
        dygraph_eval, dygraph_train = self.get_dygraph_out(input)
        static_eval, static_train = self.get_static_out(input)
        np.testing.assert_allclose(dygraph_eval.numpy(), static_eval.numpy())
        np.testing.assert_allclose(dygraph_train.numpy(), static_train.numpy())


if __name__ == "__main__":
    unittest.main()
