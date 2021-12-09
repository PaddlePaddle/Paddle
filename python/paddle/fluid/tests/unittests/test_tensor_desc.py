# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import unittest


class TestTopk(unittest.TestCase):
    def setUp(self):
        self.x = paddle.randn([10])
        self.k = 2

    def test_compile_infer(self):
        k = paddle.full([1], fill_value=self.k, dtype='int64')
        k.desc.set_desc_value([self.k])
        out, _ = paddle.topk(self.x, k)
        self.assertEqual(out.shape[0], self.k)


class TestReshape(unittest.TestCase):
    def setUp(self):
        self.shape = (2, 3, 4)
        self.x = paddle.randn(self.shape)

    def test_compile_infer(self):
        shape = paddle.shape(self.x)
        out = paddle.reshape(self.x, shape)
        self.assertEqual(shape.desc.get_content(), list(self.shape))
        self.assertEqual(out.shape, self.shape)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
