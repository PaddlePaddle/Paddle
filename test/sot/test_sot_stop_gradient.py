# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import paddle
from paddle.jit.sot import symbolic_translate


def api_with_set_stop_gradient(x):
    y = x + x
    y.stop_gradient = True
    return y


class TestApiWithSetStopGradient(unittest.TestCase):
    def test_api_with_set_stop_gradient(self):
        x_dy = paddle.to_tensor(1.0, stop_gradient=False)
        y_dy = api_with_set_stop_gradient(x_dy)
        y_dy.backward()

        x_st = paddle.to_tensor(1.0, stop_gradient=False)
        y_st = symbolic_translate(api_with_set_stop_gradient)(x_st)
        y_st.backward()

        self.assertTrue(y_dy.stop_gradient)
        self.assertTrue(y_st.stop_gradient)
        self.assertIsNone(x_dy.grad)
        self.assertIsNone(x_st.grad)


if __name__ == "__main__":
    unittest.main()
