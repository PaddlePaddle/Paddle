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
from unittest import TestCase

import paddle
from paddle import base


class TestRaiseNoDoubleGradOp(TestCase):
    def test_no_grad_op(self):
        with base.dygraph.guard():
            x = paddle.ones(shape=[2, 3, 2, 2], dtype='float32')
            x.stop_gradient = False
            y = paddle.static.nn.group_norm(x, groups=1)

            dx = base.dygraph.grad(
                outputs=[y], inputs=[x], create_graph=True, retain_graph=True
            )[0]

            loss = paddle.mean(dx)
            loss.backward()


if __name__ == '__main__':
    unittest.main()
