# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


class TestRngState(unittest.TestCase):
    def test_get_and_set_rng_state_cuda(self):
        original_state = paddle.cuda.get_rng_state()
        try:
            r = paddle.cuda.get_rng_state()
            self.assertIsInstance(r, paddle.core.GeneratorState)

            s = paddle.randn([10, 10])
            paddle.cuda.set_rng_state(r)
            s1 = paddle.randn([10, 10])
            np.testing.assert_allclose(s.numpy(), s1.numpy(), rtol=0, atol=0)
        finally:
            paddle.cuda.set_rng_state(original_state)

    def test_get_and_set_rng_state_cpu(self):
        original_state = paddle.cuda.get_rng_state('cpu')
        cur_dev = paddle.device.get_device()

        paddle.set_device('cpu')
        r = paddle.cuda.get_rng_state('cpu')
        self.assertIsInstance(r, paddle.core.GeneratorState)

        s = paddle.randn([10, 10])
        paddle.cuda.set_rng_state(r, device='cpu')
        s1 = paddle.randn([10, 10])
        np.testing.assert_allclose(s.numpy(), s1.numpy(), rtol=0, atol=0)

        paddle.cuda.set_rng_state(original_state, device='cpu')
        paddle.set_device(cur_dev)

    def test_invalid_device_raises(self):
        with self.assertRaises(ValueError):
            paddle.set_rng_state(paddle.get_rng_state(), device="unknown:0")

        original_state = paddle.get_rng_state()

        try:
            r = paddle.get_rng_state()
            if len(r) > 0:
                self.assertIsInstance(r[0], paddle.core.GeneratorState)

            s = paddle.randn([10, 10])

            paddle.set_rng_state(r)

            s1 = paddle.randn([10, 10])

            np.testing.assert_allclose(s.numpy(), s1.numpy(), rtol=0, atol=0)

        finally:
            paddle.set_rng_state(original_state)


if __name__ == "__main__":
    unittest.main()
