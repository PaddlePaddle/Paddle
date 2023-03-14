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

import paddle


class TestStopGradient(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()

    def tearDown(self):
        paddle.disable_static()

    def create_var(self, stop_gradient):
        x = paddle.randn([2, 4])
        x.stop_gradient = stop_gradient
        return x

    def test_unary(self):
        x = self.create_var(True)

        out = x.reshape([4, -1])
        self.assertTrue(out.stop_gradient)

    def test_binary(self):
        x = self.create_var(True)
        y = self.create_var(True)

        out = x + y
        self.assertTrue(out.stop_gradient)

    def test_binary2(self):
        x = self.create_var(True)
        y = self.create_var(False)

        out = x + y
        self.assertFalse(out.stop_gradient)


if __name__ == '__main__':
    unittest.main()
