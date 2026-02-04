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


class TestLSTMCellCompat(unittest.TestCase):
    def test_bias_false(self):
        cell = nn.LSTMCell(10, 20, bias=False)
        self.assertFalse(hasattr(cell, 'bias_ih'))
        self.assertFalse(hasattr(cell, 'bias_hh'))

        # Verify forward pass works without bias
        x = paddle.randn([4, 10])
        h = paddle.randn([4, 20])
        c = paddle.randn([4, 20])
        y, (new_h, new_c) = cell(x, (h, c))
        self.assertEqual(y.shape, [4, 20])
        self.assertEqual(new_h.shape, [4, 20])
        self.assertEqual(new_c.shape, [4, 20])

    def test_bias_true(self):
        cell = nn.LSTMCell(10, 20, bias=True)
        self.assertTrue(hasattr(cell, 'bias_ih'))
        self.assertTrue(hasattr(cell, 'bias_hh'))
        self.assertFalse(cell.bias_ih.stop_gradient)
        self.assertFalse(cell.bias_hh.stop_gradient)

    def test_dtype(self):
        cell = nn.LSTMCell(10, 20, dtype='float64')
        self.assertEqual(cell.weight_ih.dtype, paddle.float64)
        self.assertEqual(cell.weight_hh.dtype, paddle.float64)
        self.assertEqual(cell.bias_ih.dtype, paddle.float64)
        self.assertEqual(cell.bias_hh.dtype, paddle.float64)

        x = paddle.randn([4, 10]).astype('float64')
        h = paddle.randn([4, 20]).astype('float64')
        c = paddle.randn([4, 20]).astype('float64')
        y, (new_h, new_c) = cell(x, (h, c))
        self.assertEqual(y.dtype, paddle.float64)

    def test_device(self):
        # Only test if gpu is available, otherwise cpu
        device = 'gpu' if paddle.is_compiled_with_cuda() else 'cpu'
        cell = nn.LSTMCell(10, 20, device=device)

        # We can just check if it runs without error on the specified device
        x = paddle.randn([4, 10])
        h = paddle.randn([4, 20])
        c = paddle.randn([4, 20])
        if device == 'gpu':
            x = x.cuda()
            h = h.cuda()
            c = c.cuda()
        y, (new_h, new_c) = cell(x, (h, c))

        # Also test explicit cpu on gpu machine if possible, but 'cpu' is always safe
        cell_cpu = nn.LSTMCell(10, 20, device='cpu')
        self.assertTrue(cell_cpu.weight_ih.place.is_cpu_place())

    def test_keyword_only_args(self):
        # weight_ih_attr is keyword-only
        with self.assertRaises(TypeError):
            nn.LSTMCell(10, 20, paddle.ParamAttr())

        # This should work
        nn.LSTMCell(10, 20, weight_ih_attr=paddle.ParamAttr())


if __name__ == '__main__':
    unittest.main()
