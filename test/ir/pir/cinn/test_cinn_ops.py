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

import numpy as np

import paddle


def apply_to_static(fn, use_cinn=True):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(
        fn, build_strategy=build_strategy, full_graph=True
    )


class TestOpsBase(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        self.shape = [64, 128]
        self.axis = -1
        self.x = paddle.randn(self.shape, dtype="float32")
        self.x.stop_gradient = False
        self.y = paddle.randn(self.shape, dtype="float32")
        self.y.stop_gradient = False


class TestAddOp(TestOpsBase):
    def test_eval(self):
        self.fn = paddle.add
        cinn_out = apply_to_static(self.fn)(self.x, self.y)
        dy_out = self.fn(self.x, self.y)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


class TestIsCloseOp(TestOpsBase):
    def test_eval(self):
        self.fn = paddle.isclose
        cinn_out = apply_to_static(self.fn)(self.x, self.y)
        dy_out = self.fn(self.x, self.y)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
