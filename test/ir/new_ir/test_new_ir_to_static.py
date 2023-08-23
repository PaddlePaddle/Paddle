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

import os
import unittest

import numpy as np

import paddle

os.environ['FLAGS_enable_new_ir_api'] = 'true'  # don't work, we should


class TestDy2staticNewIR(unittest.TestCase):
    def test_basic_network(self):
        def func(x):
            paddle.mean(x)
            return x

        static_func = paddle.jit.to_static(func)
        x = paddle.randn((3, 3))
        y = paddle.randn((3, 3))
        x.stop_gradient = False
        y.stop_gradient = False
        ans = func(x)
        print("Ans: ", ans)
        print(static_func.get_concrete_program(x)[1].train_program)
        out = static_func(x)

        np.testing.assert_allclose(
            out.numpy(), ans.numpy(), rtol=1e-05, atol=1e-8
        )


if __name__ == "__main__":
    unittest.main()
