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

import unittest

import numpy as np
import utils

import paddle


class TestAddPromotion(unittest.TestCase):
    def setUp(self):
        paddle.seed(2025)
        self.prepare_info()

    def prepare_info(self):
        self.fn = paddle.add

    def check_eval(self):
        static_fn = utils.apply_to_static(self.fn, use_cinn=True)
        cinn_out = static_fn(self.x, self.y)
        dy_out = self.fn(self.x, self.y)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)

    def test_bf16_fp32(self):
        self.x = paddle.randn([3], dtype='bfloat16')
        self.y = paddle.randn([3], dtype='float32')

        self.check_eval()

    def test_f16_fp32(self):
        self.x = paddle.randn([3], dtype='float16')
        self.y = paddle.randn([3], dtype='float32')

        self.check_eval()


if __name__ == "__main__":
    unittest.main()
