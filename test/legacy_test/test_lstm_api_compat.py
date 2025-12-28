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

import paddle
from paddle import nn


class TestLSTMCompat(unittest.TestCase):
    def test_bias_false(self):
        lstm = nn.LSTM(10, 20, bias=False)
        self.assertFalse(hasattr(lstm, 'bias_ih_l0'))
        self.assertFalse(hasattr(lstm, 'bias_hh_l0'))

        # Verify forward pass works without bias
        x = paddle.randn([4, 5, 10])
        y, (h, c) = lstm(x)
        self.assertEqual(y.shape, [4, 5, 20])

    def test_bias_true(self):
        lstm = nn.LSTM(10, 20, bias=True)
        self.assertTrue(hasattr(lstm, 'bias_ih_l0'))
        self.assertTrue(hasattr(lstm, 'bias_hh_l0'))
        self.assertFalse(lstm.bias_ih_l0.stop_gradient)
        self.assertFalse(lstm.bias_hh_l0.stop_gradient)

    def test_dtype(self):
        lstm = nn.LSTM(10, 20, dtype='float64')
        self.assertEqual(lstm.weight_ih_l0.dtype, paddle.float64)
        self.assertEqual(lstm.weight_hh_l0.dtype, paddle.float64)
        self.assertEqual(lstm.bias_ih_l0.dtype, paddle.float64)
        self.assertEqual(lstm.bias_hh_l0.dtype, paddle.float64)

        x = paddle.randn([4, 5, 10]).astype('float64')
        y, (h, c) = lstm(x)
        self.assertEqual(y.dtype, paddle.float64)

    def test_device(self):
        # Only test if gpu is available, otherwise cpu
        device = 'gpu' if paddle.is_compiled_with_cuda() else 'cpu'
        lstm = nn.LSTM(10, 20, device=device)

        # We can just check if it runs without error on the specified device
        x = paddle.randn([4, 5, 10])
        if device == 'gpu':
            x = x.cuda()
        y, (h, c) = lstm(x)

        # Also test explicit cpu on gpu machine if possible, but 'cpu' is always safe
        lstm_cpu = nn.LSTM(10, 20, device='cpu')
        self.assertTrue(lstm_cpu.weight_ih_l0.place.is_cpu_place())

    def test_keyword_only_args(self):
        # direction is keyword-only
        with self.assertRaises(TypeError):
            nn.LSTM(10, 20, 1, 'forward')

        # This should work
        nn.LSTM(10, 20, 1, direction='forward')

    def test_conflict_bias_false(self):
        with self.assertRaisesRegex(ValueError, "LSTM got bias=False"):
            nn.LSTM(10, 20, bias=False, bias_ih_attr=paddle.ParamAttr())


if __name__ == '__main__':
    unittest.main()
