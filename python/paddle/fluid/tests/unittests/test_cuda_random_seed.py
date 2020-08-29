#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
"""Test cloud role maker."""

from __future__ import print_function
import os
import unittest
import paddle.fluid.generator as generator

import time  # temp for debug
import paddle.fluid as fluid
import numpy as np
import paddle
import paddle.fluid.core as core


class TestGeneratorSeed(unittest.TestCase):
    """
    Test cases for cpu generator seed.
    """

    def test_gen_dropout_dygraph(self):
        gen = paddle.manual_seed(123431)

        fluid.enable_dygraph()

        gen.manual_seed(111111111)
        st = paddle.get_cuda_state()
        x_data = np.arange(1, 101).reshape(2, 50).astype("float32")
        x = paddle.to_variable(x_data)
        #x = fluid.layers.uniform_random(
        #    [2, 10], dtype="float32", min=0.0, max=1.0)
        y = fluid.layers.dropout(x, 0.5)
        #gen.manual_seed(111111111)
        # gen.set_state(st)
        paddle.set_cuda_state(st)
        #gen = paddle.manual_seed(123431)
        #x1 = np.arange(1,101).reshape(2,50).astype("float32")
        x1 = paddle.to_variable(x_data)
        #x1 = fluid.layers.uniform_random(
        #    [2, 10], dtype="float32", min=0.0, max=1.0)
        y1 = fluid.layers.dropout(x1, 0.5)
        y_np = y.numpy()
        y1_np = y1.numpy()
        #print(y_np)
        #print(y1_np)
        if core.is_compiled_with_cuda():
            print(">>>>>>> dropout dygraph >>>>>>>")
            self.assertTrue(np.allclose(y_np, y1_np))


if __name__ == "__main__":
    unittest.main()
