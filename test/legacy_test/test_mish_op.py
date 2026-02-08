#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import unittest

import numpy as np
from op_test import get_device_place, get_places

import paddle
import paddle.base.dygraph as dg
import paddle.nn.functional as F
from paddle import nn


def ref_mish(x, threshold=20.0):
    softplus = np.select(
        [x <= threshold, x > threshold], [np.log(1 + np.exp(x)), x]
    )
    return x * np.tanh(softplus)


class TestMishOpClass_Inplace(unittest.TestCase):
    def test_case(self):
        x = np.random.uniform(-1, 1, [10, 12]).astype(np.float32)
        y_ref = ref_mish(x)

        place = get_device_place()
        with dg.guard(place) as g:
            x_var1 = paddle.to_tensor(x)
            x_var2 = paddle.to_tensor(x)

            y_var1 = F.mish(x_var1, True)
            y_test1 = y_var1.numpy()

            func = nn.Mish(True)
            y_var2 = func(x_var2)
            y_test2 = y_var2.numpy()

        np.testing.assert_allclose(y_ref, y_test1, rtol=1e-05, atol=1e-08)
        np.testing.assert_allclose(y_ref, y_test2, rtol=1e-05, atol=1e-08)

        np.testing.assert_allclose(
            y_ref, x_var1.numpy(), rtol=1e-05, atol=1e-08
        )
        np.testing.assert_allclose(
            y_ref, x_var2.numpy(), rtol=1e-05, atol=1e-08
        )


class TestMishAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.shape = [10, 12]
        self.x_np = np.random.uniform(-1, 1, self.shape).astype(np.float32)
        self.place = get_places()
        self.x_feed = copy.deepcopy(self.x_np)

    def test_api_static(self):
        paddle.enable_static()

        def run(place, inplace):
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', self.shape)
                out = F.mish(x, inplace)
                exe = paddle.static.Executor(self.place[0])
                res = exe.run(
                    feed={
                        'X': self.x_feed,
                    },
                    fetch_list=[out],
                )
            target = copy.deepcopy(self.x_np)
            out_ref = ref_mish(target)

            for out in res:
                np.testing.assert_allclose(out, out_ref, rtol=0.001)

        for place in self.place:
            run(place, True)
            run(place, False)

    def test_api_dygraph(self):
        def run(place, inplace):
            paddle.disable_static(place)
            x_tensor = paddle.to_tensor(self.x_np)
            out = F.mish(x_tensor, inplace)

            target = copy.deepcopy(self.x_np)
            out_ref = ref_mish(target)

            np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.001)

            paddle.enable_static()

        for place in self.place:
            run(place, True)
            run(place, False)


if __name__ == '__main__':
    unittest.main()
